"""
=============================================================
Module 1: YOLOv8 Spatial Vision — Object Detection
=============================================================
This module uses YOLOv8n (nano) to detect and localize physical
objects in the user's environment for collision prevention.

Architecture: Convolutional Neural Network (CNN)
Pre-trained on: COCO dataset (80 object classes)
Model variant: YOLOv8n (nano) — optimized for speed on CPU

Usage:
    python yolo_detection.py                    # Process all test images
    python yolo_detection.py --image path.jpg   # Process a single image
=============================================================
"""

import os
import sys
import time
from ultralytics import YOLO
from PIL import Image


def load_model():
    """
    Load the YOLOv8n (nano) pre-trained model.
    
    YOLOv8n is the smallest and fastest variant — ideal for
    CPU-based inference on edge/portable devices.
    The model weights are ~6MB and download automatically on first run.
    """
    print("Loading YOLOv8n model...")
    start = time.time()
    model = YOLO("yolov8n.pt")  # Auto-downloads weights on first run
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s\n")
    return model


def detect_objects(model, image_path):
    """
    Run YOLOv8 inference on a single image.
    
    The CNN processes the image through multiple convolutional layers:
    Layer 1-3: Detect edges and basic shapes
    Layer 4-8: Detect textures and patterns  
    Layer 9+:  Detect complex objects (people, cars, chairs, etc.)
    
    Args:
        model: Loaded YOLOv8 model
        image_path: Path to input image
        
    Returns:
        results: YOLOv8 results object containing detections
        detections: List of dicts with detection info
    """
    print(f"Processing: {image_path}")
    start = time.time()
    
    # Run inference — the deep neural network processes the image
    results = model(image_path, verbose=False)
    
    inference_time = time.time() - start
    
    # Parse the detections into a readable format
    detections = []
    result = results[0]  # First (and only) image result
    
    for box in result.boxes:
        detection = {
            "class_name": result.names[int(box.cls[0])],      # What object was detected
            "confidence": float(box.conf[0]),                   # How sure the model is (0-1)
            "bbox": box.xyxy[0].tolist(),                       # Bounding box coordinates [x1, y1, x2, y2]
        }
        detections.append(detection)
    
    # Print results
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Objects detected: {len(detections)}")
    
    for i, det in enumerate(detections, 1):
        conf_pct = det['confidence'] * 100
        print(f"    {i}. {det['class_name']} ({conf_pct:.1f}% confidence)")
    
    return results, detections


def save_annotated_image(results, image_path, output_dir="output"):
    """
    Save the image with bounding boxes drawn by YOLOv8.
    
    The annotated image shows:
    - Colored bounding boxes around each detected object
    - Class labels (e.g., "person", "car", "bench")
    - Confidence scores
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the annotated image (with bounding boxes drawn)
    result = results[0]
    annotated = result.plot()  # Returns a numpy array (BGR format)
    
    # Convert BGR (OpenCV format) to RGB (Pillow format) and save
    annotated_rgb = annotated[:, :, ::-1]
    img = Image.fromarray(annotated_rgb)
    
    # Save with descriptive filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"yolo_{base_name}.jpg")
    img.save(output_path, quality=95)
    
    print(f"  Saved: {output_path}\n")
    return output_path


def generate_obstacle_warning(detections):
    """
    Convert raw detections into a natural language warning message.
    
    This is what would eventually be spoken to the visually impaired user
    via the Text-to-Speech module.
    
    Example output: "Warning: 2 persons, 1 car, and 1 bench detected nearby."
    """
    if not detections:
        return "No obstacles detected. Path appears clear."
    
    # Count objects by type (only high-confidence detections >= 40%)
    object_counts = {}
    for det in detections:
        if det['confidence'] >= 0.40:
            name = det['class_name']
            object_counts[name] = object_counts.get(name, 0) + 1
    
    if not object_counts:
        return "No significant obstacles detected."
    
    # Build natural language description
    parts = []
    for obj_name, count in object_counts.items():
        if count == 1:
            parts.append(f"1 {obj_name}")
        else:
            parts.append(f"{count} {obj_name}s")
    
    if len(parts) == 1:
        objects_str = parts[0]
    elif len(parts) == 2:
        objects_str = f"{parts[0]} and {parts[1]}"
    else:
        objects_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    
    return f"Caution: {objects_str} detected nearby."


def process_all_images(input_dir="test_images", output_dir="output"):
    """
    Process all images in the test_images folder.
    Generates annotated images and obstacle warnings for each.
    """
    # Load the model once
    model = load_model()
    
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
    total_objects = 0
    
    for image_file in images:
        image_path = os.path.join(input_dir, image_file)
        
        # Run detection
        results, detections = detect_objects(model, image_path)
        
        # Save annotated image
        output_path = save_annotated_image(results, image_path, output_dir)
        
        # Generate voice warning
        warning = generate_obstacle_warning(detections)
        print(f"  Voice output: \"{warning}\"")
        print("-" * 60)
        
        total_objects += len(detections)
        all_results.append({
            "image": image_file,
            "detections": len(detections),
            "warning": warning
        })
    
    # Final summary
    print("\n" + "=" * 60)
    print("YOLO DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Images processed: {len(images)}")
    print(f"  Total objects detected: {total_objects}")
    print(f"  Average objects per image: {total_objects / len(images):.1f}")
    print(f"  Output saved to: {os.path.abspath(output_dir)}/")
    print("=" * 60)
    
    return all_results


def process_single_image(image_path, output_dir="output"):
    """Process a single image (used when --image flag is provided)."""
    model = load_model()
    results, detections = detect_objects(model, image_path)
    output_path = save_annotated_image(results, image_path, output_dir)
    warning = generate_obstacle_warning(detections)
    print(f"  Voice output: \"{warning}\"")
    return detections


class ObjectDetector:
    """
    Wrapper class for YOLOv8 object detection, providing an object-oriented
    interface compatible with the Streamlit dashboard and fusion engine.
    """
    def __init__(self, model_path="yolov8n.pt"):
        print(f"[ObjectDetector] Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        print(f"[ObjectDetector] Model loaded successfully.")

    def detect_objects(self, image, conf_threshold=0.35):
        """
        Run YOLOv8 inference on an image (numpy array or file path).

        Args:
            image: BGR numpy array or file path string.
            conf_threshold: Minimum confidence threshold (0-1).

        Returns:
            List of detection dicts with keys: class_name, confidence, conf, bbox
        """
        results = self.model(image, verbose=False, conf=conf_threshold)
        detections = []
        result = results[0]
        for box in result.boxes:
            det = {
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "conf": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            }
            detections.append(det)
        return detections


if __name__ == "__main__":
    if "--image" in sys.argv:
        # Process single image: python yolo_detection.py --image path/to/image.jpg
        idx = sys.argv.index("--image")
        if idx + 1 < len(sys.argv):
            process_single_image(sys.argv[idx + 1])
        else:
            print("Error: Please provide an image path after --image")
    else:
        # Process all images in test_images/
        process_all_images()
