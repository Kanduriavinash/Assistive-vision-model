"""
Multi-threaded resilient downloader for Salesforce/blip-image-captioning-base weights.
Resumes from the existing 256MB chunk and downloads the remainder using HTTP range requests.
"""
import os
import sys
import time
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

URL = "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin"
TOTAL_SIZE = 989820849
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks
BLOB_HASH = "d6638651a5526cc2ede56f2b5104d6851b0755816d220e5e046870430180c767"
SNAPSHOT_DIR = r"C:\Users\kandu\.cache\huggingface\hub\models--Salesforce--blip-image-captioning-base\snapshots\82a37760796d32b1411fe092ab5d4e227313294b"
BLOB_DIR = r"C:\Users\kandu\.cache\huggingface\hub\models--Salesforce--blip-image-captioning-base\blobs"
LOCAL_BLIP_DIR = r"c:\DL_PROJECT\avp\models\blip_caption"
TEMP_FILE = os.path.join(BLOB_DIR, f"{BLOB_HASH}.downloading")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(BLOB_DIR, exist_ok=True)
os.makedirs(LOCAL_BLIP_DIR, exist_ok=True)

# 1. Prepare initial data from incomplete file if present
existing_incomplete = glob.glob(os.path.join(BLOB_DIR, "*.incomplete"))
if not os.path.exists(TEMP_FILE) and existing_incomplete:
    src = existing_incomplete[0]
    print(f"Adopting existing incomplete file ({os.path.getsize(src)/(1024*1024):.1f} MB)...", flush=True)
    shutil.copyfile(src, TEMP_FILE)

curr_size = os.path.getsize(TEMP_FILE) if os.path.exists(TEMP_FILE) else 0
print(f"Resuming download from byte {curr_size:,} / {TOTAL_SIZE:,} ({curr_size/TOTAL_SIZE*100:.1f}%)", flush=True)

# Generate ranges
ranges = []
start = curr_size
while start < TOTAL_SIZE:
    end = min(start + CHUNK_SIZE - 1, TOTAL_SIZE - 1)
    ranges.append((start, end))
    start = end + 1

print(f"Total chunks to fetch: {len(ranges)} (8MB each)", flush=True)

def download_range(start_byte, end_byte, max_retries=5):
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    for attempt in range(max_retries):
        try:
            r = requests.get(URL, headers=headers, timeout=60)
            if r.status_code == 206:
                return start_byte, r.content
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {start_byte}-{end_byte}")

# If we need to append chunks sequentially
if ranges:
    t_start = time.time()
    with open(TEMP_FILE, "ab") as f_out:
        batch_size = 4
        for i in range(0, len(ranges), batch_size):
            batch = ranges[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                futures = {pool.submit(download_range, s, e): s for s, e in batch}
                results = {}
                for future in as_completed(futures):
                    s_byte, data = future.result()
                    results[s_byte] = data
            
            # Write sequentially to preserve file order
            for s, e in batch:
                f_out.write(results[s])
            f_out.flush()
            
            written = f_out.tell()
            elapsed = time.time() - t_start
            speed_mb = (written - curr_size) / (1024*1024) / max(elapsed, 0.001)
            percent = written / TOTAL_SIZE * 100
            print(f"Progress: {written/(1024*1024):.1f}/{TOTAL_SIZE/(1024*1024):.1f} MB ({percent:.1f}%) at {speed_mb:.2f} MB/s", flush=True)

# Verify size
final_size = os.path.getsize(TEMP_FILE)
if final_size == TOTAL_SIZE:
    print(f"\n[SUCCESS] Download completed! Verified size: {final_size:,} bytes", flush=True)
    # Move to blob
    blob_target = os.path.join(BLOB_DIR, BLOB_HASH)
    shutil.copyfile(TEMP_FILE, blob_target)
    # Copy to snapshot pytorch_model.bin
    snapshot_target = os.path.join(SNAPSHOT_DIR, "pytorch_model.bin")
    shutil.copyfile(TEMP_FILE, snapshot_target)
    print(f"Installed to HuggingFace snapshot: {snapshot_target}", flush=True)
    # Also copy to local project models folder for permanence
    local_target = os.path.join(LOCAL_BLIP_DIR, "pytorch_model.bin")
    if not os.path.exists(local_target):
        shutil.copyfile(TEMP_FILE, local_target)
        print(f"Installed to local project models: {local_target}", flush=True)
    
    # Clean up incomplete and temp files
    for inc in glob.glob(os.path.join(BLOB_DIR, "*.incomplete")):
        try:
            os.remove(inc)
        except Exception:
            pass
    try:
        os.remove(TEMP_FILE)
    except Exception:
        pass
    print("ALL BLIP WEIGHT FILES SUCCESSFULLY INSTALLED!", flush=True)
else:
    print(f"[ERROR] Size mismatch: got {final_size} bytes, expected {TOTAL_SIZE}", flush=True)
    sys.exit(1)
