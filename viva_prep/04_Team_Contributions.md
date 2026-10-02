# STEP 4: TEAM CONTRIBUTION MAPPING

Since your rubric specifically states that **4 out of 10 marks are for individual contribution**, the faculty *will* grade you based on how well you explain your specific part. Here is exactly what each team member must own, say, and defend.

---

## 1. VARSHA VINOD (Spatial Awareness)
**Modules Owned:** Model 1 (YOLOv8) & Model 4 (Depth Anything V2)
**Key Files:** `yolo_detection.py`, `depth_estimator.py`

### 🎤 60-Second Speaking Script:
"Good morning. I developed the spatial awareness modules of our system. My goal was to answer two questions: *What is the obstacle?* and *How close is it?* For object detection, I implemented YOLOv8. We specifically chose the 'Nano' weights (only 6 megabytes) because older models like Faster R-CNN take 2-3 seconds per frame, whereas our YOLOv8n runs at over 15 FPS on a standard CPU. However, a 2D bounding box isn't enough to guarantee safety. Therefore, I integrated Depth Anything V2—a Vision Transformer that predicts monocular depth. Without using expensive LiDAR hardware, my module generates a relative depth map that calculates if an obstacle is 5 meters away, or in critical collision range."

### 🧠 Top 3 Viva Questions for Varsha:
1. **"Why did you use YOLOv8 Nano instead of YOLOv8 Large?"**
   * *Answer:* "Because this is an assistive edge device. We cannot rely on cloud GPUs or internet connectivity for a blind user's safety. The Nano model runs fast enough on local CPU hardware to provide real-time audio warnings."
2. **"How does monocular depth estimation work without two cameras?"**
   * *Answer:* "Depth Anything V2 is a Vision Transformer that learned depth cues (like shadows, lighting, and object scale) by training on millions of images. It guesses the depth the same way the human brain does when looking at a flat 2D photograph."
3. **"What happens if YOLO detects a person, but Depth Anything fails?"**
   * *Answer:* "The bounding box will still be drawn, but the fusion engine will treat it as a general object rather than a 'Critical' proximity hazard until the depth map stabilizes."

---

## 2. MOKSHITHA YARLAGADDA (Scene Text Recognition)
**Modules Owned:** Model 2 (CRAFT) & Model 3 (CRNN)
**Key Files:** `ocr_reader.py`

### 🎤 60-Second Speaking Script:
"Following spatial detection, I developed the text perception pipeline using EasyOCR. 80% of navigational cues in the real world are text-based, like exit signs or caution boards. I implemented a two-stage neural network approach. First, Model 2 is CRAFT, which uses a VGG-16 backbone to generate heatmaps that locate *where* characters are in the image, even if the text is curved or angled. Second, Model 3 is a CRNN, which uses Bidirectional LSTMs and CTC loss to actually read the cropped text without needing to slice every individual character. This allows our system to read environmental text rapidly on CPU."

### 🧠 Top 3 Viva Questions for Mokshitha:
1. **"Why didn't you just use Pytesseract / Tesseract?"**
   * *Answer:* "Tesseract is a traditional OCR engine that struggles heavily with blurry, angled, or 'in-the-wild' camera photos. CRAFT and CRNN are deep learning models designed specifically for messy scene text."
2. **"What is the difference between CRAFT and CRNN?"**
   * *Answer:* "CRAFT is only for *Text Detection* (finding the bounding box of the word). CRNN is for *Text Recognition* (translating the picture of the word into an actual Python string)."
3. **"What is an LSTM and why is it in your OCR?"**
   * *Answer:* "Long Short-Term Memory. In CRNN, it remembers the sequence of characters. It helps the model guess a letter based on the letters that came before it, improving accuracy on blurry words."

---

## 3. AJALYA T M (Context, Fusion & Voice)
**Modules Owned:** Model 5 (BLIP), Model 6 (TTS), and the Multimodal Fusion Engine
**Key Files:** `vlm_context.py`, `multimodal_fusion_v2.py`

### 🎤 60-Second Speaking Script:
"I handled scene context and the Multimodal Fusion Layer. If Varsha's and Mokshitha's models all spoke at the exact same time, the blind user would be overwhelmed by audio clutter. I wrote the Fusion Engine to act as a traffic controller. It checks Varsha's depth maps—if an object is less than 1.5 meters away, it flags it as 'CRITICAL'. My logic guarantees that a critical collision hazard will instantly interrupt the reading of background text. Finally, I implemented the BLIP Vision-Language model for specific user queries, and the Neural TTS engine which synthesizes my prioritized strings into natural spoken audio for the user's earphones."

### 🧠 Top 3 Viva Questions for Ajalya:
1. **"What exactly is your 'Fusion Logic'?"**
   * *Answer:* "It is an algorithmic hierarchy. It extracts the X/Y coordinates from YOLO, looks up those coordinates on the Depth map, and calculates proximity. Proximity hazards strictly override OCR text in the final audio output string."
2. **"Why use BLIP instead of GPT-4?"**
   * *Answer:* "BLIP is an open-source Vision-Language Transformer that runs locally. GPT-4 requires an API key, cloud access, and high latency, which compromises the safety and privacy of our user."
3. **"How does the TTS handle rapid changes in the camera?"**
   * *Answer:* "Our UI implements an 'audio cooldown' so the TTS doesn't stutter. But if a 'CRITICAL' hazard appears, it bypasses the cooldown and interrupts the current speech."

---

## 4. AVINASH (System Architecture & Integration)
**Modules Owned:** App GUI, Mobile Server, System Lead
**Key Files:** `app_gui.py`, `mobile_api_server.py`

### 🎤 60-Second Speaking Script:
"As the system integrator, I built the architecture that connects these 6 deep learning models into a usable, real-world wearable. A major limitation of assistive tech is hardware—a blind user cannot walk around with an open laptop. I solved this by developing a local Wi-Fi API server. The user places their standard smartphone in their chest pocket, and the phone streams the camera feed silently to the laptop in their backpack. The laptop runs our heavy Fusion Engine, and streams the TTS audio back to their earphones. Finally, I developed the Perception Studio dashboard you see today, which allows us to benchmark object confidence, depth maps, and FPS latency in real-time."

### 🧠 Top 3 Viva Questions for Avinash:
1. **"How does the phone connect to the laptop without internet?"**
   * *Answer:* "Through local IPv4 sockets. The laptop hosts a FastAPI/Uvicorn server, and the phone connects via a local Wi-Fi Hotspot. The frames are sent over HTTP POST requests."
2. **"What is the overall latency (FPS) of the integrated system?"**
   * *Answer:* "By optimizing with YOLOv8 Nano and running CPU-bound inference, our loop runs in approximately X to Y FPS [check your dashboard for exact numbers, usually 1-3 FPS on heavy fusion]. While not 60FPS, it is fast enough for walking speed navigation."
3. **"What is the biggest limitation of this project?"**
   * *Answer:* "Running 6 models on a CPU creates a bottleneck. In a commercial version, we would deploy this to a dedicated Edge NPU (Neural Processing Unit) or Jetson Nano to increase the framerate and battery life."
