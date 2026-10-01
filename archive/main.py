"""
=============================================================
MAIN PIPELINE: Multimodal Assistive Vision System
=============================================================
This is the integrated pipeline that combines all three modules:
  1. YOLOv8  (CNN)  → Detects physical obstacles
  2. EasyOCR (CRNN) → Reads text from signs/labels
  3. gTTS    (TTS)  → Speaks the results as audio guidance

This is the core deliverable for the Phase 1 (50%) demo.

Usage:
    python main.py                        # Process all test images
    python main.py --image path.jpg       # Process a single image
=============================================================
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
from gtts import gTTS


# ============================================================
# MODULE 1: YOLOv8 — Spatial Object Detection (CNN)
# ============================================================

def load_yolo_model():
    """Load YOLOv8n (nano) — the lightest, fastest variant for CPU."""
    print("[YOLO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("[YOLO] Model ready.\n")
    return model


def detect_obstacles(model, image_path):
    """
    Run YOLOv8 inference to detect physical obstacles.
    Returns list of detected objects with class, confidence, and bbox.
    """
    results = model(image_path, verbose=False)
    detections = []
    result = results[0]

    for box in result.boxes:
        det = {
            "class_name": result.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),
        }
        detections.append(det)

    return results, detections


def format_obstacle_warning(detections, min_confidence=0.50):
    """Convert detections into a spoken warning sentence."""
    object_counts = {}
    for det in detections:
        if det['confidence'] >= min_confidence:
            name = det['class_name']
            object_counts[name] = object_counts.get(name, 0) + 1

    if not object_counts:
        return "No obstacles detected. Path appears clear."

    parts = []
    for obj, count in object_counts.items():
        parts.append(f"{count} {obj}" if count == 1 else f"{count} {obj}s")

    if len(parts) == 1:
        obj_str = parts[0]
    elif len(parts) == 2:
        obj_str = f"{parts[0]} and {parts[1]}"
    else:
        obj_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"Caution: {obj_str} detected nearby."


# ============================================================
# MODULE 2: EasyOCR — Text Recognition (CRNN)
# ============================================================

def load_ocr_model():
    """Load EasyOCR reader for English text."""
    print("[OCR] Loading EasyOCR model...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("[OCR] Model ready.\n")
    return reader


def read_text(reader, image_path):
    """
    Run OCR to extract text from signs, labels, documents.
    Returns list of text results with text, confidence, bbox.
    """
    results = reader.readtext(image_path)
    text_results = []
    for (bbox, text, confidence) in results:
        text_results.append({
            "text": text,
            "confidence": float(confidence),
            "bbox": bbox,
        })
    return text_results


def format_text_announcement(text_results, min_confidence=0.30):
    """Convert OCR results into a spoken announcement."""
    valid = [r['text'] for r in text_results if r['confidence'] >= min_confidence]

    if not valid:
        return "No readable text detected."

    combined = ", ".join(valid[:5])  # Limit to 5 text fragments
    return f"Text detected nearby: {combined}."


# ============================================================
# MODULE 3: gTTS — Text-to-Speech Audio Output
# ============================================================

def speak_output(message, output_path):
    """
    Convert the guidance message into an audio file.
    In a real system, this would play through the user's earpiece.
    For demo purposes, we save as an MP3 file.
    """
    try:
        tts = gTTS(text=message, lang='en', slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"  [TTS] Warning: Could not generate audio - {e}")
        return False


# ============================================================
# COMBINED PIPELINE
# ============================================================

def draw_combined_output(image_path, yolo_results, text_results, output_dir="output"):
    """
    Create a single annotated image showing BOTH:
    - YOLO bounding boxes (pink/magenta) for obstacles
    - OCR bounding boxes (green) for text regions
    """
    os.makedirs(output_dir, exist_ok=True)

    # Start with YOLO's annotated image (has pink bounding boxes)
    yolo_annotated = yolo_results[0].plot()

    # Overlay OCR text regions in green
    for result in text_results:
        if result['confidence'] < 0.20:
            continue

        bbox = result['bbox']
        text = result['text']
        confidence = result['confidence']

        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(yolo_annotated, [pts], True, (0, 255, 0), 2)

        x = int(bbox[0][0])
        y = int(bbox[0][1]) - 10
        label = f"TEXT: {text} ({confidence*100:.0f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(yolo_annotated, (x, y - th - 4), (x + tw, y + 4), (0, 255, 0), -1)
        cv2.putText(yolo_annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save combined output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"combined_{base_name}.jpg")

    annotated_rgb = yolo_annotated[:, :, ::-1]
    img = Image.fromarray(annotated_rgb)
    img.save(output_path, quality=95)

    return output_path


def process_image(image_path, yolo_model, ocr_reader, output_dir="output"):
    """
    Process a single image through the full pipeline:
    1. YOLOv8 detects obstacles
    2. EasyOCR reads text
    3. Results are combined into one guidance message
    4. gTTS converts message to audio
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    total_start = time.time()

    # --- Step 1: Object Detection ---
    yolo_start = time.time()
    yolo_results, detections = detect_obstacles(yolo_model, image_path)
    yolo_time = time.time() - yolo_start

    obstacle_msg = format_obstacle_warning(detections)
    print(f"\n  [YOLO] {len(detections)} objects found ({yolo_time:.2f}s)")
    for d in detections:
        if d['confidence'] >= 0.40:
            print(f"         - {d['class_name']} ({d['confidence']*100:.0f}%)")

    # --- Step 2: Text Recognition ---
    ocr_start = time.time()
    text_results = read_text(ocr_reader, image_path)
    ocr_time = time.time() - ocr_start

    text_msg = format_text_announcement(text_results)
    print(f"\n  [OCR]  {len(text_results)} text regions found ({ocr_time:.2f}s)")
    for t in text_results:
        if t['confidence'] >= 0.30:
            print(f"         - \"{t['text']}\" ({t['confidence']*100:.0f}%)")

    # --- Step 3: Combine into Final Guidance ---
    final_message = f"{obstacle_msg} {text_msg}"
    print(f"\n  [VOICE] Full guidance message:")
    print(f"          \"{final_message}\"")

    # --- Step 4: Save Outputs ---
    # Save combined annotated image
    img_output = draw_combined_output(image_path, yolo_results, text_results, output_dir)
    print(f"\n  [SAVE] Annotated image: {img_output}")

    # Save audio file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    audio_path = os.path.join(output_dir, f"audio_{base_name}.mp3")
    if speak_output(final_message, audio_path):
        print(f"  [SAVE] Audio guidance: {audio_path}")

    total_time = time.time() - total_start
    print(f"\n  [TIME] Total processing: {total_time:.2f}s "
          f"(YOLO: {yolo_time:.2f}s + OCR: {ocr_time:.2f}s)")

    return {
        "image": os.path.basename(image_path),
        "obstacles": len(detections),
        "texts": len(text_results),
        "message": final_message,
        "total_time": total_time,
    }


def main():
    """Main entry point — process single image or full dataset."""
    print("=" * 60)
    print("  MULTIMODAL ASSISTIVE VISION SYSTEM")
    print("  Phase 1 Demo — YOLOv8 + EasyOCR + gTTS")
    print("=" * 60)

    # Load both models
    load_start = time.time()
    yolo_model = load_yolo_model()
    ocr_reader = load_ocr_model()
    load_time = time.time() - load_start
    print(f"All models loaded in {load_time:.2f}s\n")

    # Determine input mode
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        if idx + 1 < len(sys.argv):
            result = process_image(sys.argv[idx + 1], yolo_model, ocr_reader)
        else:
            print("Error: Provide an image path after --image")
            return
    else:
        # Process all images
        input_dir = "test_images"
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        images = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ])

        if not images:
            print(f"No images found in '{input_dir}/'")
            return

        print(f"Processing {len(images)} images from '{input_dir}/'")

        all_results = []
        for img_file in images:
            img_path = os.path.join(input_dir, img_file)
            result = process_image(img_path, yolo_model, ocr_reader)
            all_results.append(result)

        # Final summary
        total_obstacles = sum(r['obstacles'] for r in all_results)
        total_texts = sum(r['texts'] for r in all_results)
        avg_time = sum(r['total_time'] for r in all_results) / len(all_results)

        print(f"\n\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"  Images processed:       {len(all_results)}")
        print(f"  Total obstacles found:  {total_obstacles}")
        print(f"  Total text regions:     {total_texts}")
        print(f"  Avg time per image:     {avg_time:.2f}s")
        print(f"  Output folder:          {os.path.abspath('output')}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
