import os
import contractions
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import emoji
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from lime.lime_text import LimeTextExplainer

# Flask app
app = Flask(__name__)
CORS(app)

nltk.download('punkt')

# ========== Preprocessing Functions ==========
def remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def remove_mentions_hashtags(text):
    return re.sub(r"@\w+|#\w+", "", text)

def preprocess_text(text):
    text = remove_urls(text)
    text = replace_emojis(text)
    text = remove_mentions_hashtags(text)
    return text.lower().strip()

# ========== Custom Layers ==========
@register_keras_serializable()
class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embedding_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        return self.position_embeddings(positions)

@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

# ========== Load Models ==========
w2v_model = Word2Vec.load("word2vec_tweet.model")
vector_size = w2v_model.vector_size

model = keras.models.load_model(
    "non_causal_tf.keras",
    custom_objects={
        "PositionEmbedding": PositionEmbedding,
        "TransformerBlock": TransformerBlock
    }
)

# ========== Embedding Utility ==========
def tokens_to_embedding(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# ========== Sentiment Labels ==========
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# ========== Prediction Function ==========
def predict_sentiment(tweet):
    cleaned = preprocess_text(tweet)
    tokens = word_tokenize(cleaned)
    embedding = tokens_to_embedding(tokens)
    input_vector = np.expand_dims(embedding, axis=(0, 1))
    prediction = model.predict(input_vector)
    sentiment_class = np.argmax(prediction, axis=1)[0]
    return sentiment_class, prediction[0]

# ========== LIME Wrapper ==========
class WrappedModel:
    def predict_proba(self, texts):
        results = []
        for text in texts:
            _, probs = predict_sentiment(text)
            results.append(probs)
        return np.array(results)

explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])

# ========== Routes ==========
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("text", "")
    if not tweet:
        return jsonify({"error": "No text provided"}), 400

    sentiment_class, probabilities = predict_sentiment(tweet)
    explanation = explainer.explain_instance(tweet, WrappedModel().predict_proba, num_features=6)

    # Format LIME explanation as list of (word, weight) tuples
    lime_weights = explanation.as_list()

    return jsonify({
        "prediction": label_map[sentiment_class],
        "probabilities": {
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]),
            "Positive": float(probabilities[2]),
        },
        "lime_explanation": lime_weights
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
