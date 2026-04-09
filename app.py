from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

def detect_qr_opencv(image):
    qr = cv2.QRCodeDetector()

    # 🔥 helper to try decode
    def try_decode(img):
        data, _, _ = qr.detectAndDecode(img)
        if data:
            return [data]

        # multi QR support
        retval, decoded_info, _, _ = qr.detectAndDecodeMulti(img)
        if retval and decoded_info:
            results = [d for d in decoded_info if d]
            if results:
                return results

        return []

    # 🔥 1. try original
    result = try_decode(image)
    if result:
        return result

    # 🔥 2. grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result = try_decode(gray)
    if result:
        return result

    # 🔥 3. contrast boost
    enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)
    result = try_decode(enhanced)
    if result:
        return result

    # 🔥 4. threshold (important for WhatsApp compression)
    _, thresh = cv2.threshold(enhanced, 140, 255, cv2.THRESH_BINARY)
    result = try_decode(thresh)
    if result:
        return result

    # 🔥 5. upscale full image
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    result = try_decode(upscaled)
    if result:
        return result

    # 🔥 6. targeted bottom-right crop (your use case)
    h, w, _ = image.shape
    crop = image[int(h*0.6):h, int(w*0.6):w]

    crop_up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    gray_crop = cv2.cvtColor(crop_up, cv2.COLOR_RGB2GRAY)
    enhanced_crop = cv2.convertScaleAbs(gray_crop, alpha=2.0, beta=0)

    for img in [crop_up, gray_crop, enhanced_crop]:
        result = try_decode(img)
        if result:
            return result

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
