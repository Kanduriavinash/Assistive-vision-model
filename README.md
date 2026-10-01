# Assistive Vision Platform (AVP)

A local, multimodal AI system designed to assist visually impaired users with real-time navigation and scene understanding. The system runs entirely on the edge (no cloud APIs required) and processes video feeds to provide audio guidance about immediate hazards, text in the environment, and general scene context.

![Dashboard UI](assets/demo_ui.png)

## Core Components
- **Object Detection:** Fine-tuned YOLOv8 for identifying everyday obstacles (doors, chairs, vehicles, stairs).
- **Depth Estimation:** Depth Anything V2 to calculate relative proximity and filter out critical hazards based on distance.
- **Text Recognition (OCR):** CRAFT + CRNN pipeline to read signs and labels in the environment.
- **Visual Question Answering:** BLIP model to answer specific user queries about the scene.
- **Mobile Integration:** A local HTTPS server that streams the user's smartphone camera back to the laptop for processing, pushing synthesized audio back to the user's earphones.

## Project Structure
- `app_gui.py`: The main Streamlit dashboard for testing and monitoring.
- `multimodal_fusion_v2.py`: The core engine that fuses bounding boxes, depth maps, and OCR to generate safe navigation prompts.
- `mobile_api_server.py` & `mobile_nav.html`: Local backend and frontend for the mobile companion app.
- `Project_Documentation/`: Contains research papers, training notebooks, and detailed architecture guides.

## Setup & Execution

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the main dashboard:**
```bash
run_phase2.bat
# or manually: streamlit run app_gui.py
```

**3. Use the mobile pocket camera:**
```bash
python mobile_api_server.py
```
*(Access the provided local IP on your phone to stream video and receive audio feedback).*
