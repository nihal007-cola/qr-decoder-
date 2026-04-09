from flask import Flask, request, jsonify
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image
import io

app = Flask(__name__)

@app.route("/decode", methods=["POST"])
def decode_qr():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')
        img_np = np.array(image)

        decoded = decode(img_np)

        if not decoded:
            return jsonify({"success": False, "data": None})

        results = [d.data.decode('utf-8') for d in decoded]

        return jsonify({
            "success": True,
            "data": results
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
