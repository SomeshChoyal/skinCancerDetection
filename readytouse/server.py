from flask import Flask, request, jsonify, render_template
from backend.predict import predict
import os
import pandas as pd
import shutil
import zipfile

app = Flask(__name__, template_folder="./frontend")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def handle_prediction():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    result = predict(image_path)  # returns {"prediction": ..., "confidence": ...}
    return jsonify(result)

@app.route("/batch_predict", methods=["POST"])
def batch_prediction():
    if "zip_folder" not in request.files or "csv_file" not in request.files:
        return jsonify({"error": "Please upload both ZIP folder and CSV file"}), 400

    # Save and extract ZIP folder
    zip_file = request.files["zip_folder"]
    zip_path = os.path.join(UPLOAD_FOLDER, "images.zip")
    extract_folder = os.path.join(UPLOAD_FOLDER, "batch_images")

    shutil.rmtree(extract_folder, ignore_errors=True)  # Clean up old images
    os.makedirs(extract_folder, exist_ok=True)

    zip_file.save(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    # Save CSV file
    csv_file = request.files["csv_file"]
    csv_path = os.path.join(UPLOAD_FOLDER, "labels.csv")
    csv_file.save(csv_path)

    # Load CSV file
    df = pd.read_csv(csv_path)

    # Iterate over images and predict
    results = []
    for image_id in df["image_id"]:
        image_path = os.path.join(extract_folder, f"{image_id}.jpg")
        if os.path.exists(image_path):
            result = predict(image_path)
            actual_label = df[df["image_id"] == image_id]["dx"].values[0]
            results.append({
                "image_id": image_id,
                "predicted": result["prediction"],
                "confidence": result["confidence"],
                "actual": actual_label
            })
        else:
            results.append({
                "image_id": image_id,
                "error": "Image not found"
            })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
