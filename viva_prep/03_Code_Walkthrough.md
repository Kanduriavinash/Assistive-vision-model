# STEP 3: CODE WALKTHROUGH (VIVA PREPARATION)

If the faculty asks you to share your screen and explain the code, here is exactly what you need to know for the most important files.

---

## 1. `yolo_detection.py` (Object Detection)
**Who owns this:** Varsha

### The Important Code:
```python
from ultralytics import YOLO

def detect_objects(model, image_path):
    # Runs the image through the neural network
    results = model(image_path, verbose=False)
    
    detections = []
    # Loop through every object the model found
    for box in results[0].boxes:
        det = {
            "label": model.names[int(box.cls[0])],  # e.g., "Chair"
            "confidence": float(box.conf[0]),       # e.g., 0.85 (85% sure)
            "bbox": box.xyxy[0].tolist()            # [x_min, y_min, x_max, y_max]
        }
        detections.append(det)
    return results, detections
```

### Viva Questions on this code:
*   **"What does `box.xyxy[0]` mean?"** 
    *   *Answer:* It stands for the X and Y coordinates of the top-left corner, and the X and Y coordinates of the bottom-right corner. It draws the literal box around the object.
*   **"What happens if you lower the confidence threshold?"**
    *   *Answer:* The model will detect more objects, but it will start making mistakes (False Positives), like thinking a shadow is a person. We keep it around 0.35 to 0.50 to balance safety and accuracy.

---

## 2. `depth_estimator.py` (Depth Anything V2)
**Who owns this:** Varsha & Mokshitha (depending on how you share it)

### The Important Code:
```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def estimate_depth(self, image):
    # Convert image for the Transformer model
    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    
    # Predict the depth
    with torch.no_grad():
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth
        
    return predicted_depth
```

### Viva Questions on this code:
*   **"Why do you use `torch.no_grad()`?"**
    *   *Answer:* It tells PyTorch *not* to calculate gradients (which are only used for training). This saves a massive amount of memory and makes the code run much faster during real-time inference (testing).
*   **"What is the output of `predicted_depth`?"**
    *   *Answer:* It is a 2D tensor (a matrix of numbers). High numbers mean the pixel is physically closer to the camera lens.

---

## 3. `ocr_reader.py` (Text Recognition)
**Who owns this:** Mokshitha

### The Important Code:
```python
import easyocr

class OCRReader:
    def __init__(self, langs=['en']):
        # gpu=False because we optimized for CPU edge devices
        self.reader = easyocr.Reader(langs, gpu=False, verbose=False)

    def extract_text(self, image):
        # Reads text and returns bounding boxes, the text, and confidence
        results = self.reader.readtext(image)
        return results
```

### Viva Questions on this code:
*   **"Why did you use EasyOCR instead of Pytesseract?"**
    *   *Answer:* Pytesseract is an older optical algorithm. EasyOCR uses deep learning (CRAFT for finding the text, and CRNN for reading it). It handles blurry, tilted, and messy real-world camera images much better.
*   **"What happens if `gpu=True`?"**
    *   *Answer:* It would run much faster if we had a dedicated NVIDIA graphics card, but we designed this to be an *edge* system that can run on a standard laptop CPU, so we set it to False.

---

## 4. `multimodal_fusion_v2.py` (The Brains / Fusion Layer)
**Who owns this:** Ajalya

### The Important Code (Simplified):
```python
def process_frame(frame, detector, depth_est, ocr_reader):
    # 1. Run YOLO, Depth, and OCR on the SAME frame
    boxes = detector.detect(frame)
    depth_map = depth_est.estimate(frame)
    text = ocr_reader.extract_text(frame)
    
    # 2. Priority Logic
    critical_objects = []
    for box in boxes:
        # Check if object is closer than 1.5 meters based on depth map
        if get_distance(box, depth_map) < 1.5:
            critical_objects.append(box.label)
            
    # 3. Audio Construction
    if len(critical_objects) > 0:
        audio_text = f"CRITICAL PROXIMITY: {critical_objects[0]} ahead."
    elif text:
        audio_text = f"Text detected: {text}"
        
    return audio_text
```

### Viva Questions on this code:
*   **"How does the fusion layer solve the 'audio clutter' problem?"**
    *   *Answer:* It uses a strict IF/ELSE hierarchy. If there is a CRITICAL physical obstacle right in front of the user, it skips reading background text so the user isn't distracted from immediate danger.
*   **"How do you get the distance of a specific object?"**
    *   *Answer:* We take the YOLO bounding box (xyxy), and we look at those exact same coordinates on the Depth Map to find the average depth value for just that object.
