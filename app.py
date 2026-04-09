from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import threading
import queue

app = Flask(__name__)

# 🔥 QUEUE SYSTEM
task_queue = queue.Queue()

def worker():
    while True:
        file = task_queue.get()
        try:
            process_image(file)
        except Exception as e:
            print("Worker error:", e)
        task_queue.task_done()

# Start background worker
threading.Thread(target=worker, daemon=True).start()

def detect_qr_opencv(image):
    qr = cv2.QRCodeDetector()

    h, w, _ = image.shape
    crop = image[int(h * 0.6):h, int(w * 0.6):w]
    crop = cv2.resize(crop, None, fx=3, fy=3)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.8)

    for img in [crop, gray, enhanced]:
        data, _, _ = qr.detectAndDecode(img)
        if data:
            print("QR FOUND:", data)
            return

    print("QR NOT FOUND")

def process_image(file):
    image = Image.open(file).convert('RGB')
    img_np = np.array(image)
    detect_qr_opencv(img_np)

@app.route("/decode", methods=["POST"])
def decode_qr():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False})

        file = request.files['file']

        # 🔥 ADD TO QUEUE (NON BLOCKING)
        task_queue.put(file.stream)

        # 🔥 INSTANT RESPONSE
        return jsonify({"success": True, "queued": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/", methods=["GET"])
def home():
    return "QR Queue Running 🚀"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
