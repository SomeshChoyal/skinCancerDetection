from flask import Flask, request, jsonify, render_template
from backend.predict import predict
import os
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

    predicted_label = predict(image_path)
    return jsonify({"prediction": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
