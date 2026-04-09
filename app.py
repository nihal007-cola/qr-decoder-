from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

def detect_qr_opencv(image):
    qr = cv2.QRCodeDetector()

    h, w, _ = image.shape

    # 🔥 precise bottom-right crop (tighter)
    crop = image[int(h * 0.65):h, int(w * 0.65):w]

    # 🔥 upscale aggressively
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # 🔥 strong contrast
    enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)

    # 🔥 threshold (VERY IMPORTANT for WhatsApp compression)
    _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)

    # Try all versions
    for img in [crop, gray, enhanced, thresh]:
        data, _, _ = qr.detectAndDecode(img)
        if data:
            return [data]

    # fallback full image
    data, _, _ = qr.detectAndDecode(image)
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
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
