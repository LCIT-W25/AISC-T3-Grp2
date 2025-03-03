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
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')
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

# Function for LIME to get model probabilities - with model-specific resizing
def predict_proba_for_lime(images, model, model_choice):
    # Normalize and prepare inputs according to model requirements
    processed_images = []
    for img in images:
        # For CNN
        if model_choice == "CNN":
            # Resize to 128x128
            img_resized = tf.image.resize(img, (128, 128)).numpy()
        # For EfficientNet
        else:
            # Resize to 224x224
            img_resized = tf.image.resize(img, (224, 224)).numpy()
        
        processed_images.append(img_resized)
    
    # Convert to batch
    processed_images = np.array(processed_images)
    
    # Get predictions
    return model.predict(processed_images)

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
    
    # Select model and size based on model choice
    if model_choice == "CNN":
        model = cnn_model
    else:
        model = efficientnet_model
    
    processed_image = preprocess_image(image, model_choice)

    try:
        # Perform inference
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        
        # For visualization (keep consistent for display)
        vis_size = (128, 128)
        
        # Create a response dictionary
        response = {
            "prediction": predicted_class,
            "confidence": f"{np.max(predictions):.2%}",
            "model_used": model_choice
        }
        
        try:
            # Generate LIME explanation
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                np.array(image.resize(vis_size)),
                lambda x: predict_proba_for_lime(x, model, model_choice),
                top_labels=5,  # Get explanations for all possible classes
                hide_color=0,
                num_samples=1000
            )
            
            # Find which labels are available in the explanation
            available_labels = explanation.top_labels
            
            # Choose the label to visualize (use predicted class if available, otherwise first available)
            if predicted_class_index in available_labels:
                label_to_explain = predicted_class_index
            elif len(available_labels) > 0:
                label_to_explain = available_labels[0]
                print(f"Warning: Using label {label_to_explain} instead of predicted class {predicted_class_index}")
            else:
                raise KeyError("No labels available in explanation")
            
            # Get the explanation mask for the chosen label
            temp, mask = explanation.get_image_and_mask(
                label=label_to_explain,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            # Save explanation image
            explanation_filename = f"lime_{model_choice}_{file.filename}.png"
            explanation_path = os.path.join(EXPLANATION_FOLDER, explanation_filename)
            
            plt.figure(figsize=(5, 5))
            plt.imshow(mark_boundaries(temp, mask))
            plt.axis("off")
            plt.title(f"LIME Explanation: {class_names[label_to_explain]}")
            plt.savefig(explanation_path)
            plt.close()
            
            response["explanation_url"] = url_for("static", filename=f"explanations/{explanation_filename}")
            
        except Exception as e:
            print(f"LIME explanation error: {str(e)}")
            print("Continuing without LIME explanation")
            # Continue without the explanation if it fails
        
        return jsonify(response)

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Detailed error in console
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)