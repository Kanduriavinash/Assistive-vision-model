"""
Deploys the GPU-trained YOLOv8 and BLIP models from vizwiz_trained_models.zip
into their production project directories.
"""
import os
import zipfile
import shutil
import glob

ZIP_PATH = r"c:\DL_PROJECT\avp\data\VIZWIZ\vizwiz_trained_models.zip"
EXTRACT_DIR = r"c:\DL_PROJECT\avp\data\VIZWIZ\extracted_models"
YOLO_TARGET_DIR = r"c:\DL_PROJECT\avp\runs\detect\runs\train\vizwiz_assistive_production"
BLIP_TARGET_DIR = r"c:\DL_PROJECT\avp\blip_vizwiz_vqa_best"

def deploy():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(os.path.join(YOLO_TARGET_DIR, "weights"), exist_ok=True)
    os.makedirs(BLIP_TARGET_DIR, exist_ok=True)
    
    print(f"Unpacking {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
        
    print("Unpack complete. Locating models and assets...")
    
    # 1. Locate and copy YOLO best.pt and evaluation charts
    best_pts = glob.glob(os.path.join(EXTRACT_DIR, "**", "best.pt"), recursive=True)
    if best_pts:
        src_best = best_pts[0]
        dest_best = os.path.join(YOLO_TARGET_DIR, "weights", "best.pt")
        shutil.copy2(src_best, dest_best)
        print(f"[SUCCESS] Deployed YOLOv8 best.pt -> {dest_best}")
        
    # Copy any png evaluation plots / results.csv
    for ext in ["*.png", "*.csv", "*.jpg"]:
        for f in glob.glob(os.path.join(EXTRACT_DIR, "**", ext), recursive=True):
            fname = os.path.basename(f)
            dest = os.path.join(YOLO_TARGET_DIR, fname)
            shutil.copy2(f, dest)
            print(f"[SUCCESS] Deployed training metric {fname} -> {dest}")
            
    # 2. Locate and copy BLIP VQA checkpoint folder
    blip_dirs = glob.glob(os.path.join(EXTRACT_DIR, "**", "blip_vizwiz_vqa_best"), recursive=True)
    if blip_dirs:
        src_blip = blip_dirs[0]
        for item in os.listdir(src_blip):
            s = os.path.join(src_blip, item)
            d = os.path.join(BLIP_TARGET_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"[SUCCESS] Deployed fine-tuned BLIP model files -> {BLIP_TARGET_DIR}")
        
    print("\n[ALL MODELS DEPLOYED SUCCESSFULLY]")

if __name__ == "__main__":
    deploy()
