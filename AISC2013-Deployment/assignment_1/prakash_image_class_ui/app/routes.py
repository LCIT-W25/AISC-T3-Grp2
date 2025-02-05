import os
from PIL import Image
from flask import render_template, request
import torch
from app import app
from .model import knn_model, label_encoder, preprocess_image, dnn_model, device


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict/knn", methods=["POST"])
def predict_knn():
    if "file" not in request.files:
        return render_template("index.html", error="No file part")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No selected file")

    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filename)

    image = Image.open(file)
    image_array = preprocess_image(image)
    prediction = knn_model.predict(image_array)

    return render_template(
        "index.html",
        prediction=prediction[0],
        image=file.filename,
    )


@app.route('/predict/dnn', methods=['POST'])
def predict_dnn():
    if "file" not in request.files:
        return render_template("index.html", error="No file part")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No selected file")

    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filename)

    image = Image.open(file)
    image_array = preprocess_image(image)

    image_tensor = torch.tensor(image_array, dtype=torch.float32).to(device)
    image_tensor = image_tensor.view(1, -1)  # Ensure batch dimension

    with torch.no_grad():
        output = dnn_model(image_tensor)
        predicted_class_idx = torch.argmax(output, dim=1).item()

    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]

    return render_template(
        "index.html",
        prediction=predicted_class,
        image=file.filename,
    )
