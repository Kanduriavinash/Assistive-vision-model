"""
=============================================================
Module 2: EasyOCR — Optical Character Recognition
=============================================================
This module uses EasyOCR to extract written text from signs,
labels, storefronts, and documents in the user's environment.

Architecture: Convolutional Recurrent Neural Network (CRNN)
  - CNN layers detect WHERE text exists in the image
  - RNN layers predict the SEQUENCE of characters
Pre-trained on: Multiple text recognition datasets

Usage:
    python ocr_reader.py                    # Process all test images
    python ocr_reader.py --image path.jpg   # Process a single image
=============================================================
"""

import os
import sys
import time
import easyocr
import cv2
import numpy as np


def load_ocr_model():
    """
    Load the EasyOCR reader model.
    
    On first run, EasyOCR downloads two neural network models:
    1. Text Detection model (CRAFT) — finds WHERE text is in the image
    2. Text Recognition model (CRNN) — reads WHAT the text says
    
    These models total ~100MB and are cached for future runs.
    """
    print("Loading EasyOCR model (English)...")
    print("(First run downloads ~100MB of model weights)\n")
    start = time.time()
    
    # Initialize reader for English text
    # gpu=False forces CPU inference (compatible with our hardware)
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    elapsed = time.time() - start
    print(f"OCR model loaded in {elapsed:.2f}s\n")
    return reader


def read_text(reader, image_path):
    """
    Run OCR on a single image to extract all visible text.
    
    The CRNN processes text in two stages:
    Stage 1 (CNN): Scans the image to find rectangular regions 
                   that likely contain text
    Stage 2 (RNN): For each text region, predicts the sequence
                   of characters from left to right
    
    Args:
        reader: EasyOCR reader object
        image_path: Path to input image
        
    Returns:
        text_results: List of dicts with text info
    """
    print(f"Processing: {image_path}")
    start = time.time()
    
    # Run OCR inference — the deep neural network processes the image
    # Returns: list of (bbox, text, confidence) tuples
    results = reader.readtext(image_path)
    
    inference_time = time.time() - start
    
    # Parse into readable format
    text_results = []
    for (bbox, text, confidence) in results:
        result = {
            "text": text,
            "confidence": float(confidence),
            "bbox": bbox,  # Four corner points of the text region
        }
        text_results.append(result)
    
    # Print results
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Text regions found: {len(text_results)}")
    
    for i, res in enumerate(text_results, 1):
        conf_pct = res['confidence'] * 100
        print(f"    {i}. \"{res['text']}\" ({conf_pct:.1f}% confidence)")
    
    return text_results


def save_annotated_image(image_path, text_results, output_dir="output"):
    """
    Save the image with text regions highlighted and extracted text overlaid.
    
    Draws:
    - Green bounding boxes around detected text regions
    - The recognized text above each region
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: Could not load image {image_path}")
        return None
    
    for result in text_results:
        bbox = result['bbox']
        text = result['text']
        confidence = result['confidence']
        
        # Only draw high-confidence detections (>= 20%)
        if confidence < 0.20:
            continue
        
        # Convert bbox points to integer coordinates
        # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        pts = np.array(bbox, dtype=np.int32)
        
        # Draw green polygon around text region
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Put the extracted text above the bounding box
        x = int(bbox[0][0])
        y = int(bbox[0][1]) - 10
        
        # Draw text background for readability
        text_label = f"{text} ({confidence*100:.0f}%)"
        (text_w, text_h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - text_h - 4), (x + text_w, y + 4), (0, 255, 0), -1)
        cv2.putText(img, text_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"ocr_{base_name}.jpg")
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"  Saved: {output_path}\n")
    return output_path


def generate_text_announcement(text_results):
    """
    Convert OCR results into a natural language announcement.
    
    This is what would be spoken to the visually impaired user
    via the Text-to-Speech module.
    
    Example: "Text detected: The sign reads 'CAFE OPEN'. Also found: 'Pull to enter'."
    """
    # Filter to high-confidence text (>= 30%)
    valid_texts = [r['text'] for r in text_results if r['confidence'] >= 0.30]
    
    if not valid_texts:
        return "No readable text detected in the scene."
    
    if len(valid_texts) == 1:
        return f"Text detected: \"{valid_texts[0]}\""
    
    # Combine multiple text fragments
    primary = valid_texts[0]
    others = valid_texts[1:]
    
    if len(others) == 1:
        return f"Text detected: \"{primary}\". Also found: \"{others[0]}\""
    else:
        other_str = "\", \"".join(others[:3])  # Limit to first 3 additional texts
        return f"Text detected: \"{primary}\". Also found: \"{other_str}\""


def process_all_images(input_dir="test_images", output_dir="output"):
    """
    Process all images in the test_images folder for text extraction.
    """
    # Load the model once
    reader = load_ocr_model()
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ])
    
    if not images:
        print(f"No images found in '{input_dir}/'")
        return
    
    print(f"Found {len(images)} images in '{input_dir}/'\n")
    print("=" * 60)
    
    all_results = []
    total_texts = 0
    images_with_text = 0
    
    for image_file in images:
        image_path = os.path.join(input_dir, image_file)
        
        # Run OCR
        text_results = read_text(reader, image_path)
        
        # Save annotated image
        save_annotated_image(image_path, text_results, output_dir)
        
        # Generate voice announcement
        announcement = generate_text_announcement(text_results)
        print(f"  Voice output: \"{announcement}\"")
        print("-" * 60)
        
        total_texts += len(text_results)
        if text_results:
            images_with_text += 1
        
        all_results.append({
            "image": image_file,
            "texts_found": len(text_results),
            "announcement": announcement
        })
    
    # Final summary
    print("\n" + "=" * 60)
    print("OCR DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Images processed: {len(images)}")
    print(f"  Images with text found: {images_with_text}")
    print(f"  Total text regions detected: {total_texts}")
    print(f"  Output saved to: {os.path.abspath(output_dir)}/")
    print("=" * 60)
    
    return all_results


def process_single_image(image_path, output_dir="output"):
    """Process a single image."""
    reader = load_ocr_model()
    text_results = read_text(reader, image_path)
    save_annotated_image(image_path, text_results, output_dir)
    announcement = generate_text_announcement(text_results)
    print(f"  Voice output: \"{announcement}\"")
    return text_results


class OCRReader:
    """
    Wrapper class for EasyOCR text recognition, providing an object-oriented
    interface compatible with the Streamlit dashboard and fusion engine.
    """
    def __init__(self, langs=None):
        if langs is None:
            langs = ['en']
        print(f"[OCRReader] Loading EasyOCR model for languages: {langs}")
        self.reader = easyocr.Reader(langs, gpu=False, verbose=False)
        print(f"[OCRReader] OCR model loaded successfully.")

    def extract_text(self, image):
        """
        Run OCR on an image to extract all visible text.

        Args:
            image: BGR numpy array or file path string.

        Returns:
            List of (bbox, text, confidence) tuples from EasyOCR.
        """
        results = self.reader.readtext(image)
        return results


if __name__ == "__main__":
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        if idx + 1 < len(sys.argv):
            process_single_image(sys.argv[idx + 1])
        else:
            print("Error: Please provide an image path after --image")
    else:
        process_all_images()
