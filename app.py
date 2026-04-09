from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

def detect_qr_opencv(image):
    qr = cv2.QRCodeDetector()
    data, bbox, _ = qr.detectAndDecode(image)

    if data:
        return [data]

    return []

@app.route("/decode", methods=["POST"])
def decode_qr():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files['file']

        image = Image.open(file.stream).convert('RGB')
        img_np = np.array(image)

        results = detect_qr_opencv(img_np)

        if not results:
            return jsonify({"success": False, "data": None})

        return jsonify({"success": True, "data": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/", methods=["GET"])
def home():
    return "QR Decoder Running 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
