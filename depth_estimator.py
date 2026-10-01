"""
=============================================================================
Multimodal Assistive Vision System - Phase 2
Module: Monocular Depth Estimation (depth_estimator.py)
-----------------------------------------------------------------------------
Performs monocular depth estimation using Depth Anything V2
via Hugging Face Transformers.
Estimates relative distances (approximate, uncalibrated) for detected objects
and generates depth colormap heatmaps for visual display.

Author: Multimodal Assistive Vision Team (Amrita University)
=============================================================================
"""

import os
import time
import numpy as np
import cv2
import torch
from PIL import Image

class DepthEstimator:
    """
    Monocular Depth Estimator powered by Depth Anything V2 / DPT Transformer.

    Note: Distances are relative estimates mapped to an approximate 0.4-8.0m range.
    Without camera calibration or depth sensor data, these are NOT true metric
    measurements. Treat all distances as rough relative proximity indicators.
    """
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device=None):
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[DepthEstimator] Initializing {self.model_name} on {self.device}...")
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"[DepthEstimator] Successfully loaded {self.model_name}.")
        except Exception as e:
            print(f"[DepthEstimator] Error loading depth model: {e}")
            raise e

    def estimate_depth(self, image_bgr):
        """
        Estimates depth map from an input BGR image.
        
        Returns:
            depth_norm (np.ndarray): Normalized depth map (0.0 to 1.0, where 1.0 is nearest).
            depth_colormap (np.ndarray): BGR colormap heatmap (INFERNO) for visualization.
            raw_depth (np.ndarray): Raw inverse depth disparity values.
        """
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        inputs = self.image_processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # Interpolate to original image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        raw_depth = prediction.cpu().numpy()
        
        # Normalize depth map to 0 - 255 for visualization
        depth_min = raw_depth.min()
        depth_max = raw_depth.max()
        
        if depth_max - depth_min > 1e-6:
            depth_norm = (raw_depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(raw_depth)
            
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        
        return depth_norm, depth_colormap, raw_depth

    def get_object_distance(self, raw_depth, bbox, min_metric=0.4, max_metric=8.0):
        """
        Estimates approximate relative distance for a detected object bounding box.

        Note: These are uncalibrated relative estimates mapped to a hand-tuned
        0.4-8.0m range from monocular depth disparity. They should be treated
        as rough proximity indicators, not true metric measurements.

        Args:
            raw_depth (np.ndarray): Raw depth disparity map.
            bbox (list/tuple): [x1, y1, x2, y2] bounding box coordinates.
            min_metric (float): Minimum distance mapping in meters (near).
            max_metric (float): Maximum distance mapping in meters (far).

        Returns:
            distance_meters (float): Estimated relative distance (approximate, uncalibrated).
            proximity_zone (str): Categorical hazard level ('CRITICAL', 'CLOSE', 'MODERATE', 'SAFE').
        """
        h, w = raw_depth.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 3.0, "MODERATE"
            
        # Sample the central 50% region of the bounding box to avoid background leakage
        box_w = x2 - x1
        box_h = y2 - y1
        cx1 = int(x1 + box_w * 0.25)
        cx2 = int(x2 - box_w * 0.25)
        cy1 = int(y1 + box_h * 0.25)
        cy2 = int(y2 - box_h * 0.25)
        
        if cx2 > cx1 and cy2 > cy1:
            roi = raw_depth[cy1:cy2, cx1:cx2]
        else:
            roi = raw_depth[y1:y2, x1:x2]
            
        # Median disparity in object ROI
        median_disp = np.median(roi)
        
        # Disparity percentile across the entire scene for robust relative scaling
        scene_min = np.percentile(raw_depth, 5)
        scene_max = np.percentile(raw_depth, 95)
        
        if scene_max - scene_min > 1e-5:
            rel_disp = np.clip((median_disp - scene_min) / (scene_max - scene_min), 0.01, 1.0)
        else:
            rel_disp = 0.5
            
        # Inverse proportional mapping to metric distance: Distance ~ 1 / Disparity
        distance_meters = round(float(min_metric + (max_metric - min_metric) * (1.0 - (rel_disp ** 0.5))), 2)
        
        # Proximity Hazard Categorization
        if distance_meters <= 1.2:
            proximity_zone = "CRITICAL"
        elif distance_meters <= 2.2:
            proximity_zone = "CLOSE"
        elif distance_meters <= 4.5:
            proximity_zone = "MODERATE"
        else:
            proximity_zone = "SAFE"
            
        return distance_meters, proximity_zone


# --- Standalone Test Execution ---
if __name__ == "__main__":
    test_img_dir = r"c:\DL_PROJECT\avp\test_images"
    out_dir = r"c:\DL_PROJECT\avp\output\depth_tests"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("Testing Monocular Depth Estimation Module (Depth Anything V2)")
    print("=" * 60)
    
    estimator = DepthEstimator()
    sample_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    for filename in sample_images:
        img_path = os.path.join(test_img_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        t0 = time.time()
        depth_norm, depth_colormap, raw_depth = estimator.estimate_depth(img)
        elapsed = (time.time() - t0) * 1000
        
        # Create side-by-side comparison
        h, w = img.shape[:2]
        depth_colored_resized = cv2.resize(depth_colormap, (w, h))
        combined = np.hstack((img, depth_colored_resized))
        
        # Add latency text
        cv2.putText(combined, f"Depth Anything V2: {elapsed:.1f}ms", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        out_path = os.path.join(out_dir, f"depth_{filename}")
        cv2.imwrite(out_path, combined)
        print(f"Processed {filename} -> {out_path} ({elapsed:.1f} ms)")
        
    print("\n[SUCCESS] Depth estimation testing completed successfully.")
