"""
Script to download and verify both BLIP captioning and BLIP VQA models.
"""
import os
import sys
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("=" * 60, flush=True)
print("[1/2] Loading BLIP Captioning Model...", flush=True)
print("=" * 60, flush=True)

from transformers import BlipProcessor, BlipForConditionalGeneration
t0 = time.time()
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
print("BlipProcessor loaded.", flush=True)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print(f"BlipForConditionalGeneration loaded in {time.time() - t0:.1f}s!", flush=True)

print("=" * 60, flush=True)
print("[2/2] Loading BLIP VQA Model...", flush=True)
print("=" * 60, flush=True)

from transformers import BlipForQuestionAnswering
t0 = time.time()
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
print("VQA Processor loaded.", flush=True)
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
print(f"BlipForQuestionAnswering loaded in {time.time() - t0:.1f}s!", flush=True)

print("=" * 60, flush=True)
print("ALL BLIP MODELS DOWNLOADED AND READY!", flush=True)
print("=" * 60, flush=True)
