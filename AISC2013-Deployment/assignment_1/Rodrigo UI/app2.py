from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import io
import pickle
import re
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

# Handle negations
def handle_negation_list(text, negation_words):
    words = text.split()
    negated = False
    processed_words = []
    
    for word in words:
        if word in negation_words:
            negated = True
            processed_words.append(word)
        elif negated:
            processed_words.append(word + "_NEG")
        else:
            processed_words.append(word)
    
    return " ".join(processed_words)

# Preprocessing
def custom_preprocessor(text):
    negation_words = ["not", "no", "never", "none", "n't", "without", "nothing"]
    text = clean_text(text)
    return handle_negation_list(text, negation_words)

# Load models and vectorizer
with open("naive_bayes_model.pkl", "rb") as nb_file:
    nb_model = pickle.load(nb_file)

with open("svm_model.pkl", "rb") as svm_file:
    svm_model = pickle.load(svm_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/Us")
def Us():
    return render_template("Us.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
        text = data.get("text", "")
        model_choice = data.get("model", "NB")  # Default to NB

        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        prediction = predict_sentiment(text, model_choice)
        return jsonify({"prediction": prediction})

    elif "file" in request.files:
        file = request.files["file"]
        model_choice = request.form.get("model", "NB")

        try:
            df = pd.read_csv(file)
        except Exception:
            return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

        if "Review" not in df.columns:
            return jsonify({"error": "CSV must contain a 'Review' column."}), 400

        reviews = df["Review"].astype(str)
        predictions = reviews.apply(lambda x: predict_sentiment(x, model_choice))

        output = io.StringIO()
        predictions.to_csv(output, header=True, index=False)
        output.seek(0)

        response = Response(output, mimetype="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response
    
    else:
        return jsonify({"error": "Invalid request"}), 400

# Predict sentiment with the selected model
def predict_sentiment(text, model_choice):
    X = vectorizer.transform([text])
    
    if model_choice == "SVM":
        prediction = svm_model.predict(X)[0]
    else:
        prediction = nb_model.predict(X)[0]

    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
