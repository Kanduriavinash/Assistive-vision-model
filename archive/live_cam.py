import cv2
import time
import threading
import queue
import subprocess
import numpy as np
from ultralytics import YOLO
import easyocr

# ============================================================
# LIVE WEBCAM ASSISTIVE VISION PIPELINE
# ============================================================

print("=" * 60)
print("  STARTING LIVE ASSISTIVE VISION SYSTEM")
print("  Press 'q' in the video window to EXIT")
print("  Press 's' to force immediate spoken guidance")
print("=" * 60)

# 1. Reliable TTS using Windows native System.Speech (via PowerShell)
#    pyttsx3.runAndWait() hangs after the first call on Windows due to
#    COM event-loop bugs. Using System.Speech.Synthesis via subprocess
#    avoids all COM threading issues — each call is a clean process.

import os
SPEAK_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speak.ps1")

speech_queue = queue.Queue()

def _tts_thread_func():
    """Dedicated thread that pulls text from the queue and speaks it
    using the speak.ps1 PowerShell helper script."""
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", SPEAK_SCRIPT, text],
                timeout=15,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        except subprocess.TimeoutExpired:
            print(f"[TTS] Speech timed out for: {text[:40]}...")
        except Exception as e:
            print(f"[TTS ERROR] {e}")

_tts_thread = threading.Thread(target=_tts_thread_func, daemon=True)
_tts_thread.start()
print("[TTS] Speech engine ready (Windows System.Speech).")

def speak_async(text):
    """Push text to the speech queue (non-blocking).
    Drops the message if the queue already has pending items
    so speech doesn't pile up and lag behind reality."""
    if speech_queue.qsize() < 2:
        speech_queue.put(text)

# 2. Load Models
print("\n[1/2] Loading YOLOv8m (Spatial Obstacle Detection)...")
yolo_model = YOLO("yolov8m.pt")  # 'm' = medium, best accuracy vs speed for assistive use

print("[2/2] Loading EasyOCR (Scene Text Recognition)...")
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

print("\n--> Connecting to Webcam (Camera Index 0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("\n[ERROR] Could not open webcam.")
    print("If you are running in a virtual environment or headless session, use 'python main.py' for image demo.")
    exit(1)

FRAME_W = 640
FRAME_H = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

last_speech_time = time.time()
last_ocr_time = 0
ocr_results = []
speech_interval = 4.0  # Speak every 4 seconds
ocr_interval = 2.0     # Run OCR every 2 seconds

# --- Helper: spatial description for a bounding box ---
def describe_position(x1, y1, x2, y2, frame_w=FRAME_W, frame_h=FRAME_H):
    """Return (direction, proximity) strings from a bounding box.
    Direction: left / right / center (based on box center x).
    Proximity: nearby / ahead / in the distance (based on box area ratio)."""
    cx = (x1 + x2) / 2
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    area_ratio = box_area / frame_area

    # Direction
    if cx < frame_w * 0.33:
        direction = "on your left"
    elif cx > frame_w * 0.66:
        direction = "on your right"
    else:
        direction = "ahead of you"

    # Proximity (bigger box = closer)
    if area_ratio > 0.15:
        proximity = "very close"
    elif area_ratio > 0.05:
        proximity = "nearby"
    else:
        proximity = "ahead"

    return direction, proximity

# --- Helper: build natural assistive speech ---
def build_assistive_speech(detections, ocr_texts):
    """Build a natural, human-friendly spoken description.
    detections: list of (name, direction, proximity) tuples.
    ocr_texts:  list of recognized text strings."""
    if not detections and not ocr_texts:
        return "The path ahead seems clear. No obstacles or text detected."

    parts = []

    # Group by object type to avoid repetition
    # e.g. 2 persons → "2 people are nearby on your left"
    from collections import defaultdict
    grouped = defaultdict(list)
    for name, direction, proximity in detections:
        grouped[name].append((direction, proximity))

    for name, instances in grouped.items():
        count = len(instances)
        # Use the closest instance for proximity description
        prox_order = {"very close": 0, "nearby": 1, "ahead": 2}
        instances.sort(key=lambda x: prox_order.get(x[1], 2))
        best_dir, best_prox = instances[0]

        # Natural phrasing
        if name == "person":
            obj_word = "person" if count == 1 else "people"
        else:
            obj_word = name if count == 1 else f"{name}s"

        if count == 1:
            parts.append(f"A {obj_word} is {best_prox} {best_dir}.")
        else:
            parts.append(f"{count} {obj_word} are {best_prox} {best_dir}.")

    # Add OCR text naturally
    if ocr_texts:
        if len(ocr_texts) == 1:
            parts.append(f"There is text that reads: {ocr_texts[0]}.")
        else:
            joined = ", ".join(ocr_texts[:3])
            parts.append(f"Text visible: {joined}.")

    # Add safety warning for close obstacles
    close_items = [name for name, _, prox in detections if prox == "very close"]
    if close_items:
        parts.insert(0, "Caution!")

    return " ".join(parts)

speak_async("Assistive Vision System Online. Camera active.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    annotated_frame = frame.copy()

    # 1. YOLOv8 Inference on Current Frame
    yolo_results = yolo_model(frame, verbose=False, conf=0.35)[0]
    
    detected_objects = []   # list of (name, direction, proximity)
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = yolo_results.names[cls_id]
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        direction, proximity = describe_position(x1, y1, x2, y2)
        detected_objects.append((name, direction, proximity))
        
        # Draw bounding box with spatial info
        color = (0, 0, 255) if proximity == "very close" else (0, 140, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {int(conf * 100)}% [{proximity}]"
        cv2.putText(annotated_frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 2. Run EasyOCR periodically (every 2s) to preserve video framerate
    if current_time - last_ocr_time > ocr_interval:
        last_ocr_time = current_time
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            raw_ocr = ocr_reader.readtext(rgb_frame)
            ocr_results = [r for r in raw_ocr if r[2] >= 0.35]
        except Exception:
            ocr_results = []

    # Draw OCR Polygons & Text
    for bbox, text, conf in ocr_results:
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        pt0 = (int(bbox[0][0]), max(18, int(bbox[0][1]) - 6))
        cv2.putText(annotated_frame, f"TXT: {text}", pt0,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 3. Formulate Natural Assistive Speech
    if current_time - last_speech_time > speech_interval:
        last_speech_time = current_time
        
        ocr_texts = [r[1] for r in ocr_results]
        full_message = build_assistive_speech(detected_objects, ocr_texts)
        print(f"[SPEECH] {full_message}")
        speak_async(full_message)

    # Top Status Banner
    cv2.rectangle(annotated_frame, (0, 0), (640, 32), (15, 23, 42), -1)
    status_text = f"Obstacles: {len(detected_objects)} | Text: {len(ocr_results)} | Press 'q' to quit"
    cv2.putText(annotated_frame, status_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Multimodal Assistive Vision - Live Camera Feed", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        last_speech_time = 0  # Trigger immediate speech

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)  # shut down the TTS thread
print("\n[INFO] Live camera session ended successfully.")
