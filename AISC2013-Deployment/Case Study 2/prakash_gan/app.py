from tensorflow.keras.models import load_model
import numpy as np
import io
import base64
import os
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS

# Load the trained generator
generator = load_model("best_generator_model.keras")

# Define noise and labels
NOISE_DIM = 128  # Noise vector size
LABEL_DIM = 5    # Number of labels in your dataset

# Define label categories (Ensure this matches your dataset)
LABELS = ["food", "drink", "inside", "outside", "menu"]

# Flask API setup
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests


def generate_image(user_label):
    """Generate an image based on a user-provided label."""
    if user_label not in LABELS:
        return None, f"⚠️ Invalid label! Choose from: {LABELS}"

    noise = np.random.normal(0, 1, (1, NOISE_DIM))  # Generate noise

    # Convert user label to one-hot encoding
    label_index = LABELS.index(user_label)
    label_one_hot = np.zeros((1, LABEL_DIM))
    label_one_hot[0, label_index] = 1  # Set the correct index to 1

    # Generate image
    generated_image = generator.predict([noise, label_one_hot])
    generated_image = (generated_image + 1) / 2  # Rescale from [-1,1] to [0,1]

    # Convert to PIL Image
    image = Image.fromarray((generated_image[0] * 255).astype(np.uint8))
    return image, None


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    user_label = data.get('label', '').strip().lower()

    # Generate image
    image, error = generate_image(user_label)
    if error:
        return jsonify({"error": error}), 400

    # Convert to base64 for sending to frontend
    img_base64 = image_to_base64(image)

    return jsonify({
        "label": user_label,
        "image": img_base64
    })


if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
