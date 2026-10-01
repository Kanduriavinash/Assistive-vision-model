"""
=============================================================================
Multimodal Assistive Vision System - Phase 2
Module: Multimodal Spatial Fusion Engine v2 (multimodal_fusion_v2.py)
-----------------------------------------------------------------------------
Fuses 2D Object Detection (YOLOv8), Monocular Relative Depth (Depth Anything V2),
Scene Text Recognition (EasyOCR), and Vision-Language Context (VLM) into
a coherent, spatial-aware assistive intelligence system.

Author: Multimodal Assistive Vision Team (Amrita University)
=============================================================================
"""

import os
import cv2
import numpy as np
from vlm_context import synthesize_navigational_narrative

class MultimodalFusionV2:
    """
    Spatial Fusion Engine combining YOLOv8, Metric Depth, OCR, and VLM.
    """
    def __init__(self):
        # Color palette for proximity hazard levels (BGR)
        self.ZONE_COLORS = {
            "CRITICAL": (0, 0, 255),      # Red (< 1.2m)
            "CLOSE": (0, 140, 255),       # Orange (1.2m - 2.2m)
            "MODERATE": (0, 220, 0),      # Green (2.2m - 4.5m)
            "SAFE": (255, 200, 0)         # Cyan/Blue (> 4.5m)
        }
        self.TEXT_COLOR = (255, 0, 255)   # Magenta for OCR Text
        
    def determine_spatial_position(self, bbox, frame_width):
        """
        Determines the horizontal sector of an object: Left, Center, or Right.
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        norm_x = center_x / frame_width
        
        if norm_x < 0.35:
            return "LEFT"
        elif norm_x > 0.65:
            return "RIGHT"
        else:
            return "CENTER"

    def fuse(self, original_img, detections, depth_norm, depth_colormap, raw_depth,
             ocr_boxes_and_text, vlm_caption=None, depth_estimator=None):
        """
        Executes end-to-end multimodal fusion.
        
        Args:
            original_img: Input BGR frame/image.
            detections: List of YOLO detections [{'bbox': [x1,y1,x2,y2], 'class_name': str, 'conf': float}]
            depth_norm: Normalized 0-1 depth map.
            depth_colormap: Colored depth heatmap (BGR).
            raw_depth: Raw disparity/depth array.
            ocr_boxes_and_text: List of tuples (bbox, text, conf) from EasyOCR.
            vlm_caption: Optional VLM scene description string.
            depth_estimator: Optional DepthEstimator instance.
            
        Returns:
            annotated_img: Rich spatial visualization image with HUD and overlays.
            fused_data: Structured dictionary of fused scene intelligence.
            audio_text: Synthesized natural language speech guidance.
        """
        h, w = original_img.shape[:2]
        canvas = original_img.copy()
        fused_objects = []
        
        # 1. Process YOLO Detections with Metric Depth
        for det in detections:
            bbox = det['bbox']
            cls_name = det.get('class_name', det.get('name', 'object'))
            conf = det.get('conf', det.get('confidence', 0.5))
            
            # Compute distance and hazard zone
            if depth_estimator is not None and raw_depth is not None:
                dist, zone = depth_estimator.get_object_distance(raw_depth, bbox)
            else:
                dist, zone = 2.5, "MODERATE"
                
            pos = self.determine_spatial_position(bbox, w)
            
            fused_objects.append({
                "label": cls_name,
                "confidence": float(conf),
                "bbox": bbox,
                "distance": dist,
                "zone": zone,
                "position": pos
            })
            
            # Draw Bounding Box with Proximity Color
            color = self.ZONE_COLORS.get(zone, (0, 255, 0))
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            
            # Label tag with Distance and Direction
            tag = f"{cls_name.upper()} ~{dist}m [{pos}]"
            if zone == "CRITICAL":
                tag = f"! HAZARD: {tag}"
                
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            # Background pill for tag
            tag_y1 = max(0, y1 - th - 8)
            cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 10, y1), color, -1)
            cv2.putText(canvas, tag, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Process OCR Scene Text
        recognized_texts = []
        for item in ocr_boxes_and_text:
            if len(item) == 3:
                t_box, text, t_conf = item
            elif len(item) == 2:
                t_box, text = item
                t_conf = 0.8
            else:
                continue
                
            if len(text.strip()) == 0 or t_conf < 0.25:
                continue
                
            recognized_texts.append(text.strip())
            
            # Draw OCR polygon/box
            if isinstance(t_box, (list, np.ndarray)) and len(t_box) >= 4:
                pts = np.array(t_box, dtype=np.int32)
                cv2.polylines(canvas, [pts], True, self.TEXT_COLOR, 2)
                # Text tag
                tx = int(pts[0][0])
                ty = max(15, int(pts[0][1]) - 5)
                cv2.putText(canvas, f"TXT: {text}", (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 2)

        # 3. Add Picture-in-Picture (PiP) Depth Map Heatmap (Top-Right)
        pip_w = int(w * 0.28)
        pip_h = int(h * 0.28)
        if depth_colormap is not None:
            depth_small = cv2.resize(depth_colormap, (pip_w, pip_h))
            cv2.rectangle(depth_small, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)
            cv2.putText(depth_small, "DEPTH HEATMAP", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            canvas[15:15 + pip_h, w - 15 - pip_w:w - 15] = depth_small

        # 4. Draw Bottom HUD / Telemetry Banner
        hud_height = 80
        hud_overlay = canvas.copy()
        cv2.rectangle(hud_overlay, (0, h - hud_height), (w, h), (20, 20, 20), -1)
        canvas = cv2.addWeighted(hud_overlay, 0.85, canvas, 0.15, 0)
        
        # Sort objects by distance (closest first)
        fused_objects.sort(key=lambda x: x['distance'])
        
        # Build HUD telemetry line 1
        if fused_objects:
            closest = fused_objects[0]
            hud_line1 = f"TARGET: {closest['label']} (~{closest['distance']}m est. {closest['position']}) | TOTAL: {len(fused_objects)} objects"
        else:
            hud_line1 = "SPATIAL SCAN: Clear pathway ahead"
            
        # Build HUD telemetry line 2
        ocr_str = f"TEXT: '{recognized_texts[0]}'" if recognized_texts else "TEXT: None"
        if vlm_caption:
            hud_line2 = f"CONTEXT: {vlm_caption[:45]}... | {ocr_str}"
        else:
            hud_line2 = f"CONTEXT: Visual Guidance Active | {ocr_str}"
            
        cv2.putText(canvas, hud_line1, (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(canvas, hud_line2, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        # 5. Generate Structured Audio Narrative
        audio_text = synthesize_navigational_narrative(
            yolo_detections=fused_objects,
            ocr_results=recognized_texts,
            depth_summary="Active",
            vlm_caption=vlm_caption
        )
        
        fused_data = {
            "objects": fused_objects,
            "texts": recognized_texts,
            "caption": vlm_caption,
            "audio_narrative": audio_text,
            "hazard_count": sum(1 for o in fused_objects if o['zone'] == 'CRITICAL')
        }
        
        return canvas, fused_data, audio_text


# --- Standalone Test Execution ---
if __name__ == "__main__":
    print("Multimodal Fusion Engine v2 initialized successfully.")
