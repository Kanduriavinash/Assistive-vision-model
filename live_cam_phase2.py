"""
=============================================================================
Multimodal Assistive Vision System - Phase 2
Module: Real-Time Live Webcam System v2 (live_cam_phase2.py)
-----------------------------------------------------------------------------
Streams live video, runs YOLOv8 + Depth Anything V2 + EasyOCR with a side-by-side
thermal depth heatmap, distance-aware spatial bounding boxes, and non-blocking
priority audio alarms.

Author: Multimodal Assistive Vision Team (Amrita University)
=============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
import subprocess
from ultralytics import YOLO
import easyocr

from depth_estimator import DepthEstimator
from multimodal_fusion_v2 import MultimodalFusionV2

class LiveVisionSystemPhase2:
    def __init__(self, camera_id=0, yolo_model="yolov8n.pt", ocr_interval=30):
        self.camera_id = camera_id
        self.ocr_interval = ocr_interval
        self.voice_enabled = True
        self.view_mode = "split" # "split" or "pip"
        
        print("=" * 70)
        print("🎥 STARTING PHASE 2 REAL-TIME MULTIMODAL LIVE VISION ENGINE")
        print("=" * 70)
        
        print("[1/3] Loading YOLOv8 Detector...")
        self.yolo = YOLO(yolo_model)
        
        print("[2/3] Loading EasyOCR Text Recognizer...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        print("[3/3] Loading Depth Estimator (Depth Anything V2)...")
        self.depth_est = DepthEstimator()
        
        self.fusion = MultimodalFusionV2()
        
        # Audio & Alert Cooldown state
        self.last_speech_time = 0
        self.speech_cooldown = 3.5 # seconds
        self.is_speaking = False
        
        self.ocr_results = []
        self.frame_count = 0
        
    def speak_async(self, text):
        """Asynchronously triggers TTS speech without blocking the video stream."""
        if not self.voice_enabled or not text or self.is_speaking:
            return
            
        current_time = time.time()
        if current_time - self.last_speech_time < self.speech_cooldown:
            return
            
        self.last_speech_time = current_time
        
        def _worker():
            self.is_speaking = True
            clean = text.replace('"', '').replace("'", "").replace('\n', ' ')
            cmd = f'PowerShell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{clean}\');"'
            try:
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            finally:
                self.is_speaking = False
                
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[Error] Could not open camera {self.camera_id}.")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "=" * 70)
        print("🚀 LIVE WEBCAM ACTIVE! Controls:")
        print("  [Q]  Quit")
        print("  [S]  Save High-Res Spatial Snapshot to output/")
        print("  [V]  Toggle Voice Speech ON / OFF")
        print("  [D]  Toggle View Mode (Split-Screen / Picture-in-Picture)")
        print("  [T]  Trigger Instant OCR Scan")
        print("=" * 70 + "\n")
        
        fps_hist = []
        
        try:
            while True:
                t_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.frame_count += 1
                
                # 1. YOLOv8 Detections
                yolo_res = self.yolo(frame, verbose=False)[0]
                detections = []
                for box in yolo_res.boxes:
                    detections.append({
                        "class_name": yolo_res.names[int(box.cls[0])],
                        "conf": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    })
                
                # 2. Depth Estimation
                depth_norm, depth_colormap, raw_depth = self.depth_est.estimate_depth(frame)
                
                # 3. EasyOCR (run periodically to preserve high FPS)
                if self.frame_count % self.ocr_interval == 1:
                    self.ocr_results = self.ocr_reader.readtext(frame)
                    
                # 4. Multimodal Spatial Fusion v2
                annotated_frame, fused_data, audio_narrative = self.fusion.fuse(
                    original_img=frame,
                    detections=detections,
                    depth_norm=depth_norm,
                    depth_colormap=depth_colormap,
                    raw_depth=raw_depth,
                    ocr_boxes_and_text=self.ocr_results,
                    vlm_caption=None,
                    depth_estimator=self.depth_est
                )
                
                # Trigger Audio Guidance if critical or closest object exists
                if fused_data['hazard_count'] > 0:
                    crit_objs = [o for o in fused_data['objects'] if o['zone'] == 'CRITICAL']
                    alert_msg = f"Caution! {crit_objs[0]['label']} is {crit_objs[0]['distance']} meters {crit_objs[0]['position']}!"
                    self.speak_async(alert_msg)
                elif fused_data['objects']:
                    c = fused_data['objects'][0]
                    self.speak_async(f"{c['label']} {c['distance']} meters on your {c['position']}")
                    
                # Compute FPS
                dt = time.time() - t_start
                fps = 1.0 / dt if dt > 0 else 30.0
                fps_hist.append(fps)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                avg_fps = np.mean(fps_hist)
                
                # Draw top status bar
                status_str = f"FPS: {avg_fps:.1f} | Voice: {'ON' if self.voice_enabled else 'OFF'} | Mode: {self.view_mode.upper()}"
                cv2.putText(annotated_frame, status_str, (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                
                # Display output based on view mode
                if self.view_mode == "split":
                    h, w = frame.shape[:2]
                    depth_resized = cv2.resize(depth_colormap, (w, h))
                    cv2.putText(depth_resized, "DEPTH HEATMAP", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    display_window = np.hstack((annotated_frame, depth_resized))
                else:
                    display_window = annotated_frame
                    
                cv2.imshow("Multimodal Assistive Vision System (Phase 2)", display_window)
                
                # Handle Keypress
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # Esc or Q
                    break
                elif key == ord('v'):
                    self.voice_enabled = not self.voice_enabled
                    print(f"[Controls] Voice Speech toggled: {'ON' if self.voice_enabled else 'OFF'}")
                elif key == ord('d'):
                    self.view_mode = "pip" if self.view_mode == "split" else "split"
                    print(f"[Controls] View Mode switched to: {self.view_mode}")
                elif key == ord('t'):
                    print("[Controls] Triggering Manual OCR Scan...")
                    self.ocr_results = self.ocr_reader.readtext(frame)
                elif key == ord('s'):
                    snap_dir = r"c:\DL_PROJECT\avp\output\snapshots"
                    os.makedirs(snap_dir, exist_ok=True)
                    snap_path = os.path.join(snap_dir, f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, display_window)
                    print(f"[Snapshot] Saved high-resolution frame -> {snap_path}")
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n[System] Live Webcam session terminated cleanly.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Webcam Multimodal Assistive Vision System (Phase 2)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--ocr_interval", type=int, default=30, help="Frames between periodic OCR scans (default: 30)")
    args = parser.parse_args()

    app = LiveVisionSystemPhase2(camera_id=args.camera, yolo_model=args.model, ocr_interval=args.ocr_interval)
    app.run()
