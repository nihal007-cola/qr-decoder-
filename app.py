from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

def try_decode(qr, img):
    # Try single decode
    data, _, _ = qr.detectAndDecode(img)
    if data:
        return [data]

    # Try multi decode (IMPORTANT)
    retval, decoded_info, _, _ = qr.detectAndDecodeMulti(img)
    if retval and decoded_info:
        return [d for d in decoded_info if d]

    return []

def detect_qr_opencv(image):
    qr = cv2.QRCodeDetector()
    h, w, _ = image.shape

    # 🔥 Define multiple regions (not just one crop)
    regions = [
        image[int(h*0.6):h, int(w*0.6):w],   # bottom-right tight
        image[int(h*0.5):h, int(w*0.5):w],   # bottom-right wider
        image[int(h*0.7):h, int(w*0.5):w],   # shifted
        image[int(h*0.5):h, int(w*0.7):w],   # shifted other way
        image  # full fallback
    ]

    for region in regions:
        try:
            # Upscale
            up = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

            # Grayscale
            gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)

            # Contrast
            enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)

            # Threshold
            _, thresh = cv2.threshold(enhanced, 140, 255, cv2.THRESH_BINARY)

            # Try all variants
            for img in [up, gray, enhanced, thresh]:
                result = try_decode(qr, img)
                if result:
                    return result

        except:
            continue

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
