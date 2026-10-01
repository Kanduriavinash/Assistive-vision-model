"""
=============================================================================
Multimodal Assistive Vision System - Phase 2
Module: Master Phase 2 Offline / Benchmark Pipeline (main_phase2.py)
-----------------------------------------------------------------------------
Executes end-to-end multimodal perception on static images & benchmarks:
YOLOv8 Object Detection + EasyOCR Scene Text + Depth Anything V2 Metric Depth +
VLM Contextual Captioning + Spatial Fusion Engine + Voice Guidance.

Author: Multimodal Assistive Vision Team (Amrita University)
=============================================================================
"""

import os
import sys
import time
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr

# Import system modules
from depth_estimator import DepthEstimator
from vlm_context import VLMContext
from multimodal_fusion_v2 import MultimodalFusionV2

def run_tts(text):
    """Speaks text aloud using Windows SAPI TTS."""
    if not text:
        return
    import subprocess
    clean_text = text.replace('"', '').replace("'", "").replace('\n', ' ')
    cmd = f'PowerShell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{clean_text}\');"'
    try:
        subprocess.Popen(cmd, shell=True)
    except Exception as e:
        print(f"[TTS] Warning: Could not trigger audio: {e}")

class MasterPipelinePhase2:
    """
    Unified Pipeline orchestrating all 4 AI perception modalities.
    """
    def __init__(self, yolo_model="yolov8n.pt", enable_vlm=True):
        print("=" * 70)
        print("🚀 MULTIMODAL ASSISTIVE VISION SYSTEM — PHASE 2 ENGINE INITIALIZATION")
        print("=" * 70)
        
        t0 = time.time()
        print("[1/4] Loading YOLOv8 Spatial Object Detector...")
        self.yolo = YOLO(yolo_model)
        
        print("[2/4] Loading EasyOCR Text Recognizer...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        print("[3/4] Loading Monocular Depth Estimator (Depth Anything V2)...")
        self.depth_est = DepthEstimator()
        
        self.enable_vlm = enable_vlm
        if self.enable_vlm:
            print("[4/4] Loading Vision-Language Context Model (BLIP)...")
            self.vlm = VLMContext()
        else:
            self.vlm = None
            
        self.fusion = MultimodalFusionV2()
        print(f"✅ All AI Engines Loaded Successfully in {time.time() - t0:.2f}s!\n")

    def process_image(self, image_path, output_dir=None, speak=False):
        """
        Processes a single image through the complete Phase 2 multimodal stack.
        """
        if not os.path.exists(image_path):
            print(f"[Error] File not found: {image_path}")
            return None
            
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Error] Could not read image: {image_path}")
            return None
            
        print(f"\n📸 Processing Image: {filename} ({img.shape[1]}x{img.shape[0]})")
        stats = {}
        
        # 1. YOLOv8 Detection
        t_yolo = time.time()
        yolo_res = self.yolo(image_path, verbose=False)[0]
        detections = []
        for box in yolo_res.boxes:
            det = {
                "class_name": yolo_res.names[int(box.cls[0])],
                "conf": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            }
            detections.append(det)
        stats['yolo_ms'] = (time.time() - t_yolo) * 1000
        
        # 2. EasyOCR Text Recognition
        t_ocr = time.time()
        ocr_results = self.ocr_reader.readtext(img)
        stats['ocr_ms'] = (time.time() - t_ocr) * 1000
        
        # 3. Depth Estimation
        t_depth = time.time()
        depth_norm, depth_colormap, raw_depth = self.depth_est.estimate_depth(img)
        stats['depth_ms'] = (time.time() - t_depth) * 1000
        
        # 4. VLM Scene Understanding
        vlm_caption = None
        if self.enable_vlm and self.vlm is not None:
            t_vlm = time.time()
            vlm_caption = self.vlm.generate_caption(img)
            stats['vlm_ms'] = (time.time() - t_vlm) * 1000
            
        # 5. Multimodal Spatial Fusion v2
        t_fuse = time.time()
        annotated_canvas, fused_data, audio_narrative = self.fusion.fuse(
            original_img=img,
            detections=detections,
            depth_norm=depth_norm,
            depth_colormap=depth_colormap,
            raw_depth=raw_depth,
            ocr_boxes_and_text=ocr_results,
            vlm_caption=vlm_caption,
            depth_estimator=self.depth_est
        )
        stats['fuse_ms'] = (time.time() - t_fuse) * 1000
        stats['total_ms'] = sum(stats.values())
        
        # Print Summary Telemetry
        print(f"  ├─ Detected Objects : {len(fused_data['objects'])} items")
        for obj in fused_data['objects']:
            print(f"  │   • {obj['label'].capitalize()}: {obj['distance']}m [{obj['position']}] (Zone: {obj['zone']}, Conf: {obj['confidence']:.2f})")
        print(f"  ├─ Recognized Text  : {', '.join(fused_data['texts']) if fused_data['texts'] else 'None'}")
        if vlm_caption:
            print(f"  ├─ VLM Description  : {vlm_caption}")
        print(f"  ├─ Audio Briefing   : \"{audio_narrative}\"")
        print(f"  └─ Latency Breakdown: YOLO: {stats['yolo_ms']:.1f}ms | Depth: {stats['depth_ms']:.1f}ms | OCR: {stats['ocr_ms']:.1f}ms | Total: {stats['total_ms']:.1f}ms")
        
        # Save Outputs
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_img_path = os.path.join(output_dir, f"phase2_{filename}")
            cv2.imwrite(out_img_path, annotated_canvas)
            
            # Also save depth colormap
            depth_out_path = os.path.join(output_dir, f"depth_{filename}")
            cv2.imwrite(depth_out_path, depth_colormap)
            
            # Save telemetry json
            json_out_path = os.path.join(output_dir, f"telemetry_{os.path.splitext(filename)[0]}.json")
            with open(json_out_path, "w") as jf:
                json.dump({
                    "image": filename,
                    "telemetry": fused_data,
                    "latency_benchmarks": stats
                }, jf, indent=2)
                
            print(f"  💾 Saved output visualization -> {out_img_path}")
            
        if speak:
            run_tts(audio_narrative)
            
        return annotated_canvas, fused_data, stats

    def process_directory(self, input_dir, output_dir):
        """Processes all images in a directory and outputs consolidated benchmark report."""
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
        
        print(f"\n📂 Starting Batch Evaluation on {len(image_files)} images from: {input_dir}")
        all_stats = []
        
        for i, fname in enumerate(image_files, 1):
            fpath = os.path.join(input_dir, fname)
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            res = self.process_image(fpath, output_dir=output_dir, speak=False)
            if res:
                all_stats.append(res[2])
                
        if all_stats:
            avg_yolo = np.mean([s['yolo_ms'] for s in all_stats])
            avg_depth = np.mean([s['depth_ms'] for s in all_stats])
            avg_ocr = np.mean([s['ocr_ms'] for s in all_stats])
            avg_total = np.mean([s['total_ms'] for s in all_stats])
            avg_fps = 1000.0 / avg_total if avg_total > 0 else 0
            
            print("\n" + "=" * 70)
            print("📊 PHASE 2 CONSOLIDATED BENCHMARK REPORT")
            print("=" * 70)
            print(f"Total Benchmark Images Evaluated : {len(all_stats)}")
            print(f"Average YOLOv8 Latency           : {avg_yolo:.1f} ms")
            print(f"Average Depth Latency            : {avg_depth:.1f} ms")
            print(f"Average EasyOCR Latency          : {avg_ocr:.1f} ms")
            print(f"Average End-to-End Latency       : {avg_total:.1f} ms ({avg_fps:.2f} FPS)")
            print(f"Results Directory                : {output_dir}")
            print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Assistive Vision System - Phase 2")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--dir", type=str, default=r"c:\DL_PROJECT\avp\test_images", help="Path to test images directory")
    parser.add_argument("--output", type=str, default=r"c:\DL_PROJECT\avp\output\phase2_eval", help="Output directory")
    parser.add_argument("--speak", action="store_true", help="Enable TTS audio feedback")
    parser.add_argument("--no_vlm", action="store_true", help="Disable VLM context")
    
    args = parser.parse_args()
    
    pipeline = MasterPipelinePhase2(yolo_model=args.model, enable_vlm=not args.no_vlm)
    
    if args.image:
        pipeline.process_image(args.image, output_dir=args.output, speak=args.speak)
    else:
        pipeline.process_directory(args.dir, output_dir=args.output)
