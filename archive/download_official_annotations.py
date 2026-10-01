"""
Downloader and extractor for official VizWiz VQA and Caption annotations.
"""
import os
import sys
import zipfile
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = r"c:\DL_PROJECT\avp\data\vizwiz"
VQA_DIR = r"c:\DL_PROJECT\avp\data\vizwiz_vqa"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VQA_DIR, exist_ok=True)

urls = [
    ("vqa_annotations.zip", "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip", VQA_DIR),
]

def download_and_extract():
    for filename, url, target_dir in urls:
        zip_path = os.path.join(target_dir, filename)
        print(f"Downloading {filename} from {url}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as out_f:
            total_size = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 65536
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB ({percent:.1f}%)", end="")
        print(f"\nDownload finished! Extracting to {target_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(target_dir)
        print(f"Extracted contents of {filename}: {os.listdir(target_dir)}")

if __name__ == "__main__":
    download_and_extract()
