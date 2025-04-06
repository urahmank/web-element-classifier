from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os
import cv2

app = Flask(__name__, template_folder="frontend", static_folder="frontend/static")
CORS(app)

model = YOLO("runs/detect/train2/weights/best.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    results = model(input_path)
    result_image = results[0].plot()

    output_path = os.path.join(OUTPUT_FOLDER, f"output_{file.filename}")
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(result_image).save(output_path)

    return send_file(output_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)