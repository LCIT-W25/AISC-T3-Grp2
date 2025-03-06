from flask import Flask, render_template, request, jsonify, url_for, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Create 'static/uploads' and 'static/explanations' folders
UPLOAD_FOLDER = "static/uploads"
EXPLANATION_FOLDER = "static/explanations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLANATION_FOLDER, exist_ok=True)

# Load models
efficientnet_model = tf.keras.models.load_model("efficientnet_model.h5")
cnn_model = tf.keras.models.load_model("cnn_model.h5")

# Define class names
class_names = ["drink", "food", "inside", "menu", "outside"]

# Image preprocessing function
def preprocess_image(image, model_choice):
    if model_choice == "CNN":
        target_size = (128, 128)  # Resize for CNN
    else:
        target_size = (224, 224)  # Resize for EfficientNet

    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# Function for LIME to get model probabilities
def predict_proba_for_lime(images, model):
    images = np.array(images)
    return model.predict(images)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    model_choice = request.form.get("model")

    if not model_choice:
        return jsonify({"error": "No model selected"}), 400

    # Save and preprocess the image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    image = Image.open(filepath).convert("RGB")
    processed_image = preprocess_image(image, model_choice)

    # Select model
    model = cnn_model if model_choice == "CNN" else efficientnet_model

    try:
        # Perform inference
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        # Generate LIME explanation
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(image.resize((128, 128))),  # Resize for visualization
            lambda x: predict_proba_for_lime(x, model),
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )

        # Get explanation mask
        temp, mask = explanation.get_image_and_mask(
            predicted_class_index,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        # Save explanation image
        explanation_filename = f"lime_{file.filename}.png"
        explanation_path = os.path.join(EXPLANATION_FOLDER, explanation_filename)

        plt.figure(figsize=(5, 5))
        plt.imshow(mark_boundaries(temp, mask))
        plt.axis("off")
        plt.title(f"LIME Explanation: {predicted_class}")
        plt.savefig(explanation_path)
        plt.close()

        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{np.max(predictions):.2%}",
            "model_used": model_choice,
            "explanation_url": url_for("static", filename=f"explanations/{explanation_filename}")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
