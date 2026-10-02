# STEP 6: VIVA QUESTION BANK

Here are the most likely questions your panel will ask, grouped by category, with the exact answers you should give.

## Basic & Project Justification
1. **What is the main objective of this project?** To provide visually impaired users with real-time, audio-based spatial and text awareness using a single mobile camera.
2. **Why didn't you just use an ultrasonic sensor (like a smart cane)?** Ultrasonic sensors only tell you *distance*, not *what* the object is. Our system tells the user if the obstacle is a person or a bicycle.
3. **Why use both Object Detection and OCR?** Object detection finds physical obstacles to avoid, while OCR reads signs (like "Exit") to help with actual navigation and context.
4. **Who is the target audience?** Completely blind or severely visually impaired individuals navigating both indoor and outdoor environments.
5. **How does the user interact with the system?** They wear their smartphone in a chest pocket (camera facing out) and listen to audio feedback through earphones.

## Technical & Architecture (The 6 Models)
6. **Which Object Detection model did you use?** YOLOv8 (specifically the Nano variant).
7. **Why YOLOv8 over Faster R-CNN?** YOLO is a single-stage detector. It looks at the image once, making it incredibly fast (real-time). Faster R-CNN is too slow for walking speeds.
8. **Why the 'Nano' version of YOLO?** It is only ~6MB and runs efficiently on a CPU, which is crucial since our edge system doesn't rely on massive cloud GPUs.
9. **How do you calculate distance without two cameras (stereo vision)?** We use Depth Anything V2, a Vision Transformer that performs *monocular* depth estimation by analyzing shadows, lighting, and scale from a single 2D image.
10. **What is EasyOCR?** A deep learning-based text reader that handles messy, angled "in-the-wild" text better than older software like Tesseract.
11. **What are the two parts of EasyOCR?** CRAFT (for detecting where the text is) and CRNN (for reading the characters).
12. **What does BLIP do in your project?** It is a Vision-Language Model used for Visual Question Answering (VQA). If the user wants context (e.g., "What color is the car?"), BLIP answers it.
13. **Why use BLIP instead of GPT-4?** Privacy, cost, and offline safety. BLIP runs entirely locally, meaning the user isn't stranded if their Wi-Fi drops.
14. **How do you convert the text to audio?** We use a Neural Text-to-Speech (TTS) engine that directly accesses the device's native voice synthesis.
15. **What is the Multimodal Fusion Layer?** It is our custom algorithm that combines the outputs of YOLO, Depth, and OCR, and prioritizes them so the user doesn't hear overlapping audio.

## Code-Level & Implementation
16. **What language and framework did you use?** Python and PyTorch.
17. **What is a Bounding Box?** A rectangular box drawn around a detected object, defined by X and Y coordinates (min and max).
18. **What does `confidence_threshold = 0.35` mean?** The model must be at least 35% sure that an object is what it thinks it is before reporting it.
19. **What happens if you set the threshold to 0.05?** We would get "False Positives"—the system would hallucinate objects that aren't there, like thinking a shadow is a person.
20. **What happens if you set the threshold to 0.95?** "False Negatives"—the system would miss real objects unless the lighting is perfect, which is dangerous for the user.
21. **How does the mobile phone communicate with the laptop?** The laptop runs a FastAPI/Uvicorn HTTP server. The phone sends frames via POST requests over a local Wi-Fi Hotspot.
22. **Why didn't you run the Python code directly on the phone?** iOS and Android restrict heavy Python PyTorch execution. The laptop acts as an edge processing hub in the user's backpack.
23. **What is `torch.no_grad()`?** A PyTorch command used during inference (testing) that disables gradient calculation to save memory and increase speed.
24. **How do you extract depth for a specific object?** We take the bounding box coordinates from YOLO and slice that exact same region out of the Depth Map tensor to find the average pixel intensity.
25. **What does the 'Cooldown' variable do in your audio script?** It prevents the TTS from repeating "Chair, Chair, Chair" every single frame, giving the user 4 seconds of silence between identical warnings.

## Limitations & Critical Thinking
26. **What is the biggest limitation of your system?** The COCO dataset limitation. YOLO only knows 80 classes, so it cannot identify "walls" or "doors", only generic objects like chairs or people.
27. **What is another limitation?** Processing 6 models on a standard laptop CPU causes a bottleneck, reducing our Frames Per Second (FPS).
28. **How does poor lighting affect the system?** Standard RGB cameras fail in the dark, meaning YOLO and OCR will fail.
29. **What happens if the Wi-Fi connection drops?** The phone cannot send frames to the laptop, and the system halts.
30. **How would you solve the Wi-Fi issue?** By using a direct USB-C tether from a wearable camera to a Raspberry Pi or Jetson Nano, removing wireless networking entirely.
31. **Does this replace a white cane?** No. This is a *complementary* system. The cane handles immediate ground-level drop-offs (like stairs), while our AI handles waist-level and head-level context.
32. **Is it safe to use earphones for a blind person?** We recommend bone-conduction headphones so the user's ears remain open to hear real-world traffic.

## Future Scope
33. **How will you fix the "Wall / Door" detection problem?** In Phase 2, we will implement "Global Depth Thresholding." If the Depth Map detects any large object within 0.5 meters, it will warn the user of a "Physical Barrier" even if YOLO doesn't classify it.
34. **How could you improve the framerate (FPS)?** By deploying the code to specialized edge hardware with an NPU (Neural Processing Unit), like the Nvidia Jetson Orin Nano.
35. **What are your plans for the VizWiz dataset?** We plan to train YOLO further on VizWiz to recognize blind-specific items like medicine bottles and white canes.
36. **Could you add GPS to this?** Yes. Future iterations could fuse Google Maps API data so the system says "Turn left at the STOP sign."

*(Note: These 36 highly-curated questions combine the concepts of over 60 standard Viva questions into the most impactful, defense-ready format).*
