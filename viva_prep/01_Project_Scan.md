# STEP 1: PROJECT SCAN & FOUNDATION

## 1. One-Paragraph Plain-English Summary
Imagine a visually impaired person trying to walk across a college campus. A walking cane can only tell them if they bump into a wall, but it can't tell them if a bicycle is speeding toward them, or read a sign that says "Caution: Wet Floor." Our project is a **Multimodal Assistive Vision System** that acts as a pair of artificial eyes. Using a smartphone camera placed in their shirt pocket, the system constantly scans the environment using deep learning. It detects physical obstacles, calculates exactly how far away they are, reads text on signs, and understands the scene's context. Finally, it acts like a smart human guide, filtering out useless noise and speaking only the most critical, life-saving warnings directly into their earphones in real-time.

## 2. Full Folder & File Structure
Here are the exact core files in your project and what they do:

- **`app_gui.py`**: The main user interface. It runs a beautiful Streamlit dashboard where you can see the live camera feed, adjust settings, and monitor what the AI is detecting.
- **`mobile_api_server.py`**: The wireless bridge. It runs a local backend server that allows the user's smartphone to stream video over Wi-Fi to the laptop for processing.
- **`mobile_nav.html`**: The frontend webpage that the user opens on their phone to access the camera and receive audio.
- **`multimodal_fusion_v2.py`**: **(The Brains)** The core fusion engine. It combines the data from all the AI models, prioritizes the most dangerous hazards, and writes the English sentence to be spoken.
- **`yolo_detection.py`**: Runs YOLOv8 to draw bounding boxes around physical objects (chairs, doors, people).
- **`depth_estimator.py`**: Runs *Depth Anything V2* to look at the 2D image and calculate the 3D distance of obstacles in meters.
- **`ocr_reader.py`**: Runs EasyOCR (CRAFT + CRNN) to find and read text on signs or documents.
- **`vlm_context.py`**: Runs the BLIP Transformer to answer complex visual questions (VQA).
- **`speak.ps1`**: A PowerShell script that forces Windows to synthesize text into spoken audio without needing an internet connection.
- **`run_phase2.bat`**: A simple double-click shortcut to launch the whole project.

## 3. The Tech Stack (What & Why)

| Technology / Model | What it does | WHY we chose it over alternatives |
| :--- | :--- | :--- |
| **Python & PyTorch** | Core programming and Deep Learning framework. | Industry standard, massive community, and direct support for all our models. |
| **YOLOv8 Nano (Ultralytics)** | Object Detection (Model 1) | We used the **Nano (~6.2MB)** weights. Older models (Faster R-CNN) take 2-3 seconds per frame. YOLOv8n runs in ~100ms on a CPU, which is mandatory for real-time safety. |
| **Depth Anything V2 (HuggingFace)** | Depth Estimation (Model 4) | It provides incredible monocular depth (guessing depth from a single flat lens). This avoids forcing the blind user to wear heavy, expensive LiDAR or dual-stereo cameras. |
| **EasyOCR (CRAFT + CRNN)** | Text Reading (Models 2 & 3) | Tesseract is older and struggles with curved/blurry signs. EasyOCR uses deep learning (VGG-16 backbone) which is far more robust for messy "in-the-wild" camera angles. |
| **BLIP (Salesforce)** | Visual Q&A (Model 5) | GPT-4 Vision is a cloud API. If the Wi-Fi drops, a blind person loses their eyes. BLIP runs 100% locally on edge hardware, ensuring total privacy and safety. |
| **Streamlit** | Dashboard UI | Much faster to build than React/Django, perfect for ML evaluation dashboards. |
| **FastAPI / Uvicorn** | Mobile Server (`mobile_api_server.py`) | Extremely fast async Python server to handle live video frame streaming over local Wi-Fi without lag. |

## 4. How to Run & Demo
**To Run:**
1. Open a terminal and run `.\venv\Scripts\activate`
2. Run `streamlit run app_gui.py` to open the dashboard.
3. In a second terminal, run `python mobile_api_server.py` to start the phone connection.

**Demo Strategy:**
1. Open the dashboard on the projector.
2. Have Avinash connect his phone to the laptop's Mobile Hotspot, open the IP address in his phone browser, and click "Start".
3. Point the phone at a team member -> System says "Person detected".
4. Point the phone at a printed piece of paper that says "EXIT" -> System says "Text detected: EXIT".
5. Walk the phone very close to a chair -> System interrupts and shouts "CRITICAL PROXIMITY - CHAIR".
