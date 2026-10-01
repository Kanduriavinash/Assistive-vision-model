# 👁️ MULTIMODAL ASSISTIVE VISION SYSTEM FOR VISUALLY IMPAIRED NAVIGATION
## Definitive Project Documentation & Master Guide

> **Course:** 23CSE473 Neural Networks and Deep Learning  
> **Institution:** Amrita Vishwa Vidyapeetham  
> **Team Members:** Kanduri Venkata Sai Mani Avinash, Varsha Vinod, Mokshitha Yarlagadda, Ajalya T M  

---

## 📋 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Deep Learning System Architecture (The 6 Models)](#2-deep-learning-system-architecture-the-6-models)
3. [The Multimodal Spatial & Priority Fusion Engine](#3-the-multimodal-spatial--priority-fusion-engine)
4. [Dataset & Empirical Training (VizWiz Benchmark)](#4-dataset--empirical-training-vizwiz-benchmark)
5. [Operational Modes & Software Interfaces](#5-operational-modes--software-interfaces)
6. [Team Presentation Script & 10-Minute Slide Allocation](#6-team-presentation-script--10-minute-slide-allocation)
7. [Viva Defense: Tough Professor Questions & Model Answers](#7-viva-defense-tough-professor-questions--model-answers)
8. [Codebase Directory & File Reference](#8-codebase-directory--file-reference)

---

# 1. EXECUTIVE SUMMARY & PROBLEM STATEMENT

### 🌍 The Human Challenge
Over **2.2 Billion people worldwide** live with vision impairment. Navigating unfamiliar indoor and outdoor environments presents severe daily hazards:
* **Physical Obstacles:** Stairs, open doors, low-hanging signs, parked bikes, and moving vehicles.
* **Text Blindness:** Over **80% of critical navigation cues** (bus route numbers, room numbers, caution boards, product labels) are visual text that traditional aids cannot interpret.

### 🚫 Why Existing Tools Fail
| Traditional Aid | Operating Principle | Why It Fails |
| :--- | :--- | :--- |
| 🦯 **White Cane** | Physical ground touch | Only detects objects upon contact at ground level; cannot detect head-height hazards or read text. |
| 📡 **Ultrasonic Smart Cane** | Acoustic echo beeping | A generic "beep" cannot tell *what* an object is (a harmless curtain vs. an open manhole). |
| 🐕 **Guide Dogs** | Animal training | Expensive ($50,000+), requires years of training, cannot read text signage. |
| 📱 **Be My Eyes App** | Sighted volunteer video call | Requires human volunteer availability, continuous high-speed internet, and invades user privacy. |

### 💡 Our Solution
An end-to-end edge AI assistive system that integrates **6 Deep Learning Models** with a **Multimodal Spatial Priority Fusion Engine** to:
1. **Detect** physical obstacles and navigation hazards in real-time.
2. **Estimate** relative depth and proximity zones (*Critical*, *Close*, *Safe*).
3. **Read** scene text from signs, storefronts, and boards.
4. **Understand** holistic scene context and answer natural language user queries (VQA).
5. **Synthesize** prioritized, natural spoken voice guidance delivered to the user's earpiece.

---

# 2. DEEP LEARNING SYSTEM ARCHITECTURE (THE 6 MODELS)

```
📸 Input Camera Image
   ├── Model 1: YOLOv8 Nano (CNN) ───────────────► 2D Bounding Boxes & Classes
   ├── Model 2: CRAFT (CNN Heatmap) ─────────────► Text Region Localization
   │    └── Model 3: CRNN (BiLSTM + CTC) ────────► Text Character Transcription
   ├── Model 4: Depth Anything V2 (ViT) ────────► Dense Relative Depth Map
   ├── Model 5: BLIP Transformer (VLM) ──────────► Scene Caption & VQA Answers
   │
   └──► 🧠 MULTIMODAL SPATIAL & PRIORITY FUSION LAYER
        │
        └──► Model 6: Neural TTS (Speech Engine) ─► 🎧 Spoken Voice Guidance
```

### Detailed Breakdown of the 6 Deep Learning Models:

| # | Model Name | Neural Architecture | Primary Task | Key Parameters / Loss Function |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **YOLOv8 Nano** | CNN (CSPDarknet + FPN + Decoupled Head) | Real-time 2D Obstacle & Hazard Detection | 3.01M params, CIoU Bounding Box + BCE Class Loss |
| **2** | **CRAFT** | Deep CNN (VGG-16 Backbone) | Scene Text Detection via Gaussian Heatmaps | Character Region Awareness & Affinity Loss |
| **3** | **CRNN** | CNN + Bidirectional LSTM + CTC | Sequence-to-Sequence Text Recognition | Connectionist Temporal Classification (CTC) Loss |
| **4** | **Depth Anything V2** | Vision Transformer (DINOv2 ViT + DPT) | Monocular Relative Depth Estimation | Scale-Shift Invariant (SSI) Depth Loss |
| **5** | **BLIP Transformer** | Multimodal Vision-Language Transformer | Visual Question Answering (VQA) & Context | Cross-Entropy Next-Token Generation Loss |
| **6** | **Neural TTS** | Acoustic Neural Synthesizer (gTTS / SAPI) | Voice Guidance Audio Generation | Text $\rightarrow$ Audio Waveform Conversion |

---

# 3. THE MULTIMODAL SPATIAL & PRIORITY FUSION ENGINE

### The Audio Clutter Problem
If an assistive system detects 10 objects and 20 text snippets simultaneously and speaks everything, the result is acoustic chaos:
> *"person person chair chair car table bus stop exit wet floor open monday..."*

### Our Smart Arbitration Solution (`multimodal_fusion_v2.py`)
1. **Confidence Filtering:** Filters detections (Objects $\ge 35\%$, Text $\ge 30\%$).
2. **Spatial Distance Zoning:**
   * **🔴 Critical Zone ($< 1.5\text{m}$):** Immediate collision hazard (interrupts speech queue).
   * **🟠 Navigational Zone ($1.5\text{m} - 3.5\text{m}$):** Directional guidance (*"ahead of you"*, *"on your left"*).
   * **🟢 Background Zone ($> 3.5\text{m}$):** Filtered out to maintain silence.
3. **Quantity Aggregation:** Converts duplicates (*"chair, chair, chair"* $\rightarrow$ *"3 chairs"*).
4. **Natural Language Synthesis:** Formats clean, actionable guidance:
   > *"Caution! Immediate obstacle: 1 chair approximately 1.1 meters on your left. Sign reads: 'EXIT'."*

---

# 4. DATASET & EMPIRICAL TRAINING (VIZWIZ BENCHMARK)

### Why the VizWiz Benchmark?
Standard datasets (COCO, ImageNet) feature sharp, well-lit, centered photos taken by sighted people. Real blind users capture images with **motion blur**, **extreme lighting variations**, **tilt**, and **finger occlusions**.

### Dataset Partition in Repository:
* **Train Split:** 23,954 images (`data/vizwiz/train/`)
* **Validation Split:** 7,750 images (`data/vizwiz/val/`)
* **Test Split:** 8,000 images (`data/vizwiz/test/`)
* **VizWiz-VQA Dataset:** 28,523 Ground-Truth Questions & Answers (`data/vizwiz_vqa/`)

### Empirical Training Results (YOLOv8 Assistive Fine-Tuning):
* **Checkpoint:** `runs/detect/runs/train/vizwiz_assistive_production/weights/best.pt`
* **Hazard Recall:** **95.1%** (+31.8% boost over vanilla COCO)
* **Hazard mAP@50:** **0.909**
* **Inference Speed:** **92.5 ms (CPU)** / **8.2 ms (GPU)**

---

# 5. OPERATIONAL MODES & SOFTWARE INTERFACES

### 1. Interactive Streamlit Web Studio (`app_gui.py`)
Run command:
```powershell
.\venv\Scripts\streamlit.exe run app_gui.py
```
* **Mode 1 (Photo Studio):** Upload any image or choose benchmark scenes to see YOLO detections, Depth colormap, OCR text, and speech.
* **Mode 2 (Live Camera Scanner):** Live webcam feed with depth analysis and auto-speech.
* **Mode 3 (VizWiz VQA Assistant):** Browse 4,300+ authentic blind-user photos and ask natural language questions answered by BLIP.
* **Mode 4 (Benchmark Dashboard):** Interactive loss progression and confusion matrix viewer.

### 2. Full Offline Batch Pipeline (`main_phase2.py`)
```powershell
.\venv\Scripts\python.exe main_phase2.py --dir test_images --output output/phase2_eval
```

### 3. Real-Time OpenCV Live Webcam (`live_cam_phase2.py`)
```powershell
.\venv\Scripts\python.exe live_cam_phase2.py
```
* **Keyboard Controls:** `Q` (Quit), `S` (Save Snapshot), `V` (Mute/Unmute Speech), `D` (Toggle Depth View), `T` (Force OCR Scan).

---

# 6. TEAM PRESENTATION SCRIPT & 10-MINUTE SLIDE ALLOCATION

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       10-MINUTE TEAM TALKING PLAN                           │
├───────────────────┬───────────────────────────────────┬─────────────────────┤
│ 1. Avinash        │ Problem Statement & Overview      │ 0:00 - 2:30 (2.5m)  │
│ 2. Varsha         │ Model 1 (YOLOv8) & Model 4 (Depth)│ 2:30 - 4:30 (2.0m)  │
│ 3. Mokshitha      │ Model 2 (CRAFT) & Model 3 (CRNN)  │ 4:30 - 6:30 (2.0m)  │
│ 4. Ajalya         │ Model 5 (BLIP), Model 6 & Fusion  │ 6:30 - 8:30 (2.0m)  │
│ 5. Avinash (Wrap) │ Demo, VizWiz Results & Conclusion │ 8:30 - 10:00 (1.5m) │
└───────────────────┴───────────────────────────────────┴─────────────────────┘
```

### Cue Cards for Each Member:

* **Avinash (Lead Presenter):**
  * *Opening:* "Respected professors and colleagues, 2.2 Billion people live with vision impairment. A cane alerts you upon collision, but cannot read 'WET FLOOR'. Our system provides real-time vision, depth, and speech."
  * *Handover:* "I hand over to Varsha to explain spatial object detection and depth estimation."

* **Varsha Vinod:**
  * *Key Points:* YOLOv8 single-stage CNN architecture (CSPDarknet + FPN), 3.01M parameters, sub-100ms latency. Depth Anything V2 monocular Vision Transformer for proximity zoning without stereo cameras.
  * *Handover:* "Next, Mokshitha will explain scene text reading."

* **Mokshitha Yarlagadda:**
  * *Key Points:* 80% of cues are text. CRAFT (Model 2) uses 2D Gaussian heatmaps to locate character regions. CRNN (Model 3) uses CNN + BiLSTM + CTC Loss to transcribe words without per-character slicing.
  * *Handover:* "Ajalya will now present the VLM, Speech engine, and Multimodal Fusion Layer."

* **Ajalya T M:**
  * *Key Points:* BLIP Transformer (Model 5) for holistic scene context and VQA. Neural TTS (Model 6) for audio speech. Multimodal Fusion Layer arbitration logic (3 distance zones, duplicate suppression, collision priority queue).
  * *Handover:* "Back to Avinash to present our live demo and benchmark metrics."

---

# 7. VIVA DEFENSE: TOUGH PROFESSOR QUESTIONS & MODEL ANSWERS

### ❓ Q1: "Why YOLOv8 and not Faster R-CNN or SSD?"
> **Answer:** "Faster R-CNN is a two-stage detector using a Region Proposal Network (RPN). On CPU, it takes 2.0 to 3.0 seconds per frame, which is too slow for walking safety. YOLOv8 is a single-stage anchor-free CNN executing in ~92ms on CPU and 8ms on GPU with 3.0M parameters, making real-time edge execution feasible."

### ❓ Q2: "How does CRNN read words without segmenting characters?"
> **Answer:** "CRNN uses **CTC (Connectionist Temporal Classification) Loss**. The CNN extracts feature maps, the Bidirectional LSTM predicts character probabilities across vertical image slices, and the CTC layer merges duplicate characters and blanks (e.g., `b-b-u-s-s` $\rightarrow$ `bus`) without manual bounding boxes per letter."

### ❓ Q3: "How does Depth Anything V2 calculate distance without LiDAR?"
> **Answer:** "Depth Anything V2 is a **Monocular Relative Depth Estimator** trained on 62 million images with a Vision Transformer (DINOv2). It learns geometric visual priors (linear perspective, scale variation, occlusion edges) to output dense inverse-depth $\frac{1}{d}$. We map this into discrete safety proximity zones: *Critical (<1.5m)*, *Close (1.5–3.5m)*, and *Safe*."

### ❓ Q4: "What is your novel contribution if you are using existing architectures?"
> **Answer:** "Our novel contribution is threefold:
> 1. **The Multimodal Spatial Priority Fusion Engine** that resolves the audio clutter dilemma through distance zoning and priority queueing.
> 2. **Domain Adaptation on the VizWiz Benchmark** adapting models to authentic blind-user photography yielding **95.1% hazard recall**.
> 3. **An integrated edge pipeline** combining 2D detection, depth estimation, OCR, VQA, and speech into one unified assistive application."

---

# 8. CODEBASE DIRECTORY & FILE REFERENCE

```
c:\DL_PROJECT\avp\
├── app_gui.py                      # Interactive Streamlit Web Studio (All 4 Modes)
├── main_phase2.py                  # Offline Phase 2 Multimodal Batch Perception Runner
├── live_cam_phase2.py              # Real-Time OpenCV Live Webcam with Depth Overlay
├── yolo_detection.py               # Model 1: YOLOv8 Object & Hazard Detector
├── ocr_reader.py                   # Models 2 & 3: EasyOCR (CRAFT + CRNN) Text Reader
├── depth_estimator.py              # Model 4: Depth Anything V2 Relative Depth Engine
├── vlm_context.py                  # Model 5: BLIP Vision-Language Transformer & VQA
├── multimodal_fusion_v2.py         # Custom Multimodal Spatial & Priority Fusion Engine
├── prepare_and_train_vizwiz.py     # Production Local VizWiz Training Script
├── train_vizwiz_colab.ipynb        # Dual-Model GPU Google Colab Trainer (YOLOv8 + BLIP)
├── PROJECT_MASTER_STUDY_GUIDE.html # Interactive HTML Presentation Study Guide
├── data/vizwiz/                    # 39,704 Real VizWiz Images (train, val, test)
├── data/vizwiz_vqa/                # 28,523 Ground-Truth VQA Annotations (JSON)
└── runs/detect/runs/train/         # Production Trained Weights (best.pt) & Metric Plots
```
