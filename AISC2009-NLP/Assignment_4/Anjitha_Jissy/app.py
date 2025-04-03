from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer
model_path = "model"  # Path to your saved model folder
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")  # Base model
model = PeftModel.from_pretrained(base_model, model_path)

# Move model to CPU
device = "cpu"
model.to(device)
model.eval()

@app.route("/", methods=["GET"])
def home():
    return "Flask app is running! Use /generate endpoint to interact with the model."

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Get input JSON
        data = request.get_json()
        prompt = data.get("prompt", "")
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate text
        output = model.generate(input_ids, max_length=100, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
