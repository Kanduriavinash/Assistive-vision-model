"""
=============================================================================
Multimodal Assistive Vision System - Phase 2
Module: Mobile Navigation API Server (mobile_api_server.py)
-----------------------------------------------------------------------------
Lightweight HTTP server that:
1. Serves the mobile_nav.html page to the phone browser
2. Accepts base64-encoded JPEG frames via POST /api/process_frame
3. Runs YOLOv8 + Depth + OCR + Fusion on each frame
4. Returns JSON with {narrative, objects, hazard_level}

Usage:
    python mobile_api_server.py
    Then open http://<your-ip>:8502 on the phone (or scan the QR code)

Author: Multimodal Assistive Vision Team (Amrita University)
=============================================================================
"""

import os
import sys
import json
import base64
import time
import threading
import io
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Load AI Models (one-time)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  AVP Mobile Navigation API Server")
print("=" * 60)
print()

from yolo_detection import ObjectDetector
from ocr_reader import OCRReader
from depth_estimator import DepthEstimator
from multimodal_fusion_v2 import MultimodalFusionV2

detector = ObjectDetector()
ocr_engine = OCRReader()
depth_est = DepthEstimator()
fusion = MultimodalFusionV2()

try:
    from vlm_context import VLMContext
    vlm = VLMContext(load_transformer=True)
    print("[VLMContext] BLIP loaded successfully.")
except Exception as e:
    print(f"[VLMContext] Warning: Could not load VLM: {e}")
    vlm = None


# OCR caching (run every Nth request to save latency)
_ocr_cache = {"texts": [], "count": 0}
OCR_EVERY_N = 3

# ─────────────────────────────────────────────────────────────
# Process a single frame
# ─────────────────────────────────────────────────────────────

def process_vqa(base64_jpeg, question):
    try:
        img_bytes = base64.b64decode(base64_jpeg)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"answer": "Could not decode image."}
        if vlm:
            ans = vlm.answer_question(frame, question)
            return {"answer": ans}
        else:
            return {"answer": "BLIP model is not loaded on the server."}
    except Exception as e:
        return {"answer": f"Error: {e}"}

def process_frame(base64_jpeg):
    """Decode base64 JPEG, run full perception pipeline, return result dict."""
    try:
        img_bytes = base64.b64decode(base64_jpeg)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"narrative": "Could not decode image.", "objects": [], "hazard_level": "safe"}
    except Exception as e:
        return {"narrative": f"Image decode error: {e}", "objects": [], "hazard_level": "safe"}

    t0 = time.time()

    # 1. Object Detection
    detections = detector.detect_objects(frame, conf_threshold=0.30)

    # 2. Periodic OCR
    _ocr_cache["count"] += 1
    if _ocr_cache["count"] % OCR_EVERY_N == 1:
        _ocr_cache["texts"] = ocr_engine.extract_text(frame)

    # 3. Depth Estimation
    depth_norm, depth_colormap, raw_depth = depth_est.estimate_depth(frame)

    # 4. Multimodal Fusion
    annotated_img, fused_data, audio_narrative = fusion.fuse(
        original_img=frame,
        detections=detections,
        depth_norm=depth_norm,
        depth_colormap=depth_colormap,
        raw_depth=raw_depth,
        ocr_boxes_and_text=_ocr_cache["texts"],
        vlm_caption=None,
        depth_estimator=depth_est
    )

    elapsed_ms = (time.time() - t0) * 1000

    # Determine hazard level
    hazard_level = "safe"
    obj_list = []
    for obj in fused_data.get("objects", []):
        zone = obj.get("zone", "SAFE")
        obj_list.append({
            "label": obj.get("label", ""),
            "distance": obj.get("distance", "?"),
            "position": obj.get("position", "center"),
            "zone": zone
        })
        if zone == "CRITICAL":
            hazard_level = "critical"
        elif zone == "CLOSE" and hazard_level != "critical":
            hazard_level = "close"

    return {
        "narrative": audio_narrative or "Path appears clear.",
        "objects": obj_list,
        "hazard_level": hazard_level,
        "latency_ms": round(elapsed_ms, 1)
    }


# ─────────────────────────────────────────────────────────────
# HTTP Request Handler
# ─────────────────────────────────────────────────────────────
MOBILE_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobile_nav.html")

class NavigationHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress default access logs for cleanliness."""
        pass

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path in ("/", "/mobile_nav.html", "/index.html"):
            try:
                with open(MOBILE_HTML_PATH, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self._set_headers(404, "text/plain")
                self.wfile.write(b"mobile_nav.html not found")
        else:
            self._set_headers(404, "text/plain")
            self.wfile.write(b"Not found")

    def do_POST(self):
        if self.path == "/api/process_frame":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                base64_img = data.get("image", "")
                result = process_frame(base64_img)
                self._set_headers(200)
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        elif self.path == "/api/vqa":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                base64_img = data.get("image", "")
                question = data.get("question", "What is in front of me?")
                result = process_vqa(base64_img, question)
                self._set_headers(200)
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))


# ─────────────────────────────────────────────────────────────
# Server Entry Point
# ─────────────────────────────────────────────────────────────
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    import ssl
    port = 8502
    server = HTTPServer(("0.0.0.0", port), NavigationHandler)
    local_ip = get_local_ip()

    # Wrap with SSL for HTTPS (required for camera access on mobile browsers)
    cert_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.crt")
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.key")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        protocol = "https"
    else:
        protocol = "http"
        print("[WARN] SSL cert not found. Camera may not work on phone browser.")

    print()
    print("=" * 60)
    print(f"  AVP Mobile Navigation Server ACTIVE")
    print(f"  Local:   {protocol}://localhost:{port}")
    print(f"  Network: {protocol}://{local_ip}:{port}")
    print()
    print(f"  Open the Network URL on your phone browser.")
    print(f"  (Accept the security warning - it's a self-signed cert)")
    print("=" * 60)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
