# STEP 5: PRESENTATION PREP & SLIDE OUTLINE

Here is the exact structure you should use for your PPT slides, followed by a bulletproof script for your live demo.

---

## 1. Slide-by-Slide Outline (Matches the Rubric)

**Slide 1: Title Slide**
*   Project Title: Multimodal Assistive Vision System for the Visually Impaired
*   Team Members: Varsha, Ajalya, Mokshitha, Avinash

**Slide 2: Problem Statement**
*   Visually impaired people lack complete environmental context.
*   Existing tools only do *one* thing (either beep at obstacles OR read text), but they don't combine them.
*   *Goal:* Create a unified system that detects objects, calculates depth, reads text, and speaks actionable guidance.

**Slide 3: System Architecture (Required by Rubric)**
*   *Visual:* Put the Mermaid Flowchart from Step 2 here.
*   *Explanation:* Show the Smartphone Camera -> Wi-Fi -> Laptop processing (YOLO + Depth + OCR) -> Fusion Engine -> Audio out.

**Slide 4: Methodology - Spatial Awareness**
*   Mention YOLOv8 Nano for lightweight, real-time object detection.
*   Mention Depth Anything V2 for monocular depth estimation without expensive LiDAR.

**Slide 5: Methodology - Text Perception**
*   Mention EasyOCR.
*   Explain the 2-step process: CRAFT (finds the text) + CRNN (reads the text).

**Slide 6: Methodology - Multimodal Fusion**
*   Explain the "Traffic Cop" logic.
*   How the system calculates distance and creates a "CRITICAL" override to prevent audio overlap.

**Slide 7: Results & Performance (Required by Rubric)**
*   Show screenshots of the Perception Studio dashboard.
*   Mention your latency: The system runs entirely on Edge CPU hardware without requiring cloud GPUs, proving its feasibility as a real-world wearable.

**Slide 8: Limitations & Future Scope**
*   *Limitation:* CPU bottleneck causes lower framerates.
*   *Future Scope:* Porting the system to a dedicated Jetson Nano or Google Coral board for 60 FPS performance.

**Slide 9: Conclusion & Demo Transition**
*   Summary of success.
*   "We will now demonstrate the Live Navigation system."

---

## 2. The 3-Minute Live Demo Script

**Preparation (Before the meeting starts):**
1. Ensure your laptop is connected to your Mobile Hotspot.
2. Ensure `mobile_api_server.py` is running in the background.
3. Open `https://<YOUR_IP>:8502` on your phone browser. Keep it ready.
4. Have a printed piece of paper with the word "EXIT" written in big letters.

**The Script (Avinash speaks):**
*"For our demonstration, we will show you our system working in real-time. Since a blind user cannot carry an open laptop, we have built a local Wi-Fi bridging API. I am placing my smartphone in my shirt pocket, and it is acting as the user's camera."*

**(Action: Avinash clicks "START" on his phone browser).**

*"The phone is now streaming frames silently to the Fusion Engine running on this laptop. I will now walk toward Varsha."*

**(Action: Avinash points the phone at Varsha).**
*System speaks:* **"Person detected."**

*"As you can hear, the YOLOv8 model detected a person. Now, I will hold up a sign."*

**(Action: Avinash holds up the 'EXIT' sign in front of the camera).**
*System speaks:* **"Text detected: EXIT."**

*"The EasyOCR pipeline successfully read the environmental text. Finally, we will demonstrate the Fusion Engine's safety override. I will walk extremely close to this chair."*

**(Action: Avinash moves the phone very close to a chair).**
*System speaks:* **"CRITICAL PROXIMITY - CHAIR!"**

*"Because the Depth Anything V2 model registered the pixels as being less than 1.5 meters away, the Fusion Engine tagged it as a CRITICAL HAZARD and immediately alerted the user. This concludes our end-to-end demonstration."*
