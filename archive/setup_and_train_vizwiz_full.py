"""
=============================================================================
Multimodal Assistive Vision System - Phase 2
Module: Full VizWiz Dataset Setup & Automated Trainer (setup_and_train_vizwiz_full.py)
-----------------------------------------------------------------------------
Automates the extraction, formatting, and full multi-epoch training pipeline
for the official VizWiz dataset (Train, Validation, and Test splits).

Author: Multimodal Assistive Vision Team (Amrita University)
=============================================================================
"""

import os
import sys
import zipfile
import argparse
import time
from pathlib import Path
from ultralytics import YOLO

VIZWIZ_ROOT = r"c:\DL_PROJECT\avp\data\vizwiz"

def extract_zip_if_found(zip_path, target_dir):
    """Extracts a ZIP archive into target directory if it exists."""
    if os.path.exists(zip_path):
        print(f"[VizWiz] Unpacking {zip_path} -> {target_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"[VizWiz] Successfully extracted {os.path.basename(zip_path)}!")
        return True
    return False

def verify_and_prepare_vizwiz_structure(base_dir=VIZWIZ_ROOT):
    """Ensures complete train, val, and test directory structure."""
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(base_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", s), exist_ok=True)
        
    yaml_path = os.path.join(base_dir, "vizwiz_full.yaml")
    yaml_text = f"""
# =====================================================================
# Official VizWiz Assistive Navigation Dataset Configuration
# =====================================================================
path: {os.path.abspath(base_dir)}
train: images/train
val: images/val
test: images/test

# Assistive Hazard & Object Class Taxonomy
names:
  0: person
  1: chair
  2: door
  3: stairs
  4: table
  5: vehicle
  6: sign
  7: hazard
  8: cup_bottle
  9: crosswalk
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_text.strip())
        
    print(f"[VizWiz] Master dataset configuration created at: {yaml_path}")
    return yaml_path

def start_full_training(yaml_path, epochs=25, batch=8, imgsz=640, device=""):
    """Launches full epoch training using YOLOv8."""
    print("=" * 70)
    print("🚀 LAUNCHING FULL VIZWIZ MULTI-EPOCH TRAINING")
    print(f"Dataset YAML : {yaml_path}")
    print(f"Epochs       : {epochs}")
    print(f"Batch Size   : {batch}")
    print(f"Image Size   : {imgsz}x{imgsz}")
    print("=" * 70)
    
    # Load starting checkpoint
    model = YOLO("yolov8n.pt")
    
    t0 = time.time()
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device if device else None,
            project="runs/train",
            name="vizwiz_full_production",
            patience=15,
            save=True,
            val=True,
            verbose=True
        )
        elapsed = time.time() - t0
        print(f"\n✅ [VizWiz] Full training pipeline completed in {elapsed/60:.2f} minutes!")
        print(f"Best model weights saved to: runs/train/vizwiz_full_production/weights/best.pt")
        return results
    except Exception as e:
        print(f"[VizWiz] Training encountered error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full VizWiz Dataset Trainer")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU")
    
    args = parser.parse_args()
    
    # Verify/create structure
    yaml_config = verify_and_prepare_vizwiz_structure()
    
    # Check if user has zip files in data/vizwiz/
    for z in ["train.zip", "val.zip", "test.zip", "Annotations.zip"]:
        z_path = os.path.join(VIZWIZ_ROOT, z)
        extract_zip_if_found(z_path, VIZWIZ_ROOT)
        
    start_full_training(yaml_config, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)
