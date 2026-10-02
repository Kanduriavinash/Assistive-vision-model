# STEP 7: ML / CV CONCEPT REFRESHER

If the faculty asks you to explain the theoretical math or definitions behind your AI, use this cheat sheet. Explain them exactly as written here—they are formatted to sound academic but easy to understand.

## 1. Deep Learning & Neural Networks
*   **CNN (Convolutional Neural Network):** An AI algorithm that is specifically designed to look at images. Instead of looking at every pixel at once, it uses "filters" (like a magnifying glass) to scan the image for edges, corners, and eventually full shapes.
*   **Transformer (ViT - Vision Transformer):** A newer type of AI (used in your Depth model). Instead of scanning pixel-by-pixel like a CNN, it breaks the image into "patches" and figures out how every patch relates to every other patch. This is how it understands global context (like shadows and lighting) to guess depth.
*   **Epoch:** One complete pass of the training dataset through the neural network during the training phase.
*   **Inference:** The testing phase. This is what your live system does. It is taking a pre-trained model and asking it to predict live data.

## 2. Object Detection (YOLOv8)
*   **YOLO (You Only Look Once):** Traditional models (R-CNN) look at an image thousands of times to find objects. YOLO divides the image into a grid and predicts all bounding boxes and classes in *one single pass*. This makes it incredibly fast.
*   **Bounding Box:** The X, Y coordinates that draw a rectangle around an object.
*   **Confidence Score:** A percentage (e.g., 0.85) representing how mathematically certain the AI is that its prediction is correct.
*   **IoU (Intersection over Union):** A metric used to evaluate accuracy. It measures how perfectly the AI's bounding box overlaps with the real object's actual location.
*   **NMS (Non-Maximum Suppression):** Sometimes YOLO accidentally draws 5 boxes around the exact same chair. NMS is an algorithm that looks at overlapping boxes and deletes all of them except the one with the highest confidence score.

## 3. Depth & Spatial
*   **Monocular Depth Estimation:** Calculating the 3D distance of objects using only a single, flat 2D camera lens. It works by teaching the AI to understand perspective, object sizes (a tiny car is far away), and shadows.
*   **LiDAR:** A laser radar that shoots lasers to measure exact depth. *We DO NOT use this because it is too expensive and heavy for our users.* We use Monocular Depth instead.

## 4. Text & Context
*   **OCR (Optical Character Recognition):** The process of turning pixels of letters into actual computer string text.
*   **Heatmap (CRAFT):** The CRAFT model outputs a heatmap—a color-coded image where "hot" areas (red) indicate a very high probability that a text character is located there.
*   **LSTM (Long Short-Term Memory):** Used in CRNN. It is a type of network that has "memory". It helps the AI read words because it remembers the letter it just read. If it reads "H-E-L-L", the LSTM knows the next letter is likely "O".
*   **VLM (Vision-Language Model):** An architecture (like BLIP) that combines a Vision model (to look at an image) and an NLP model (to process text). This allows it to answer written questions about a photograph.

## 5. Metrics & Evaluation
*   **True Positive (TP):** AI says it's a Chair, and it IS a Chair. (Good).
*   **False Positive (FP):** AI says it's a Chair, but it's actually a shadow. (Bad).
*   **False Negative (FN):** There is a Chair, but the AI completely misses it. (Very Bad for a blind person).
*   **mAP (Mean Average Precision):** The standard academic score (from 0 to 100) used to grade how accurate an object detection model is across all of its classes.
