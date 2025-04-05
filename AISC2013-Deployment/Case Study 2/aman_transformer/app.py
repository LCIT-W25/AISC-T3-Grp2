

import os
import re
import numpy as np
import emoji
import contractions
import nltk
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Flask app
app = Flask(__name__)
CORS(app)

# Load Word2Vec model
w2v_model = Word2Vec.load("word2vec_tweet.model")
tokenizer = Tokenizer()


class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        return self.position_embeddings(positions)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


# Load the trained Transformer model
with custom_object_scope({'TransformerBlock': TransformerBlock, 'PositionEmbedding': PositionEmbedding}):
    model = load_model("non_causal_tf.keras")


sentiment_pipeline = pipeline("sentiment-analysis")

# Combined preprocessing function
slang_dict = {"u": "you", "ur": "your", "idk": "i don't know",
              "btw": "by the way", "smh": "shaking my head"}


def preprocess_input(text):
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\\S+|www\\S+|https\\S+", '', text)
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
    text = text.lower()
    text = ' '.join([slang_dict.get(word, word) for word in text.split()])
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens


def tweet_to_embedding(tokens, embed_dim=100):
    embeddings = []
    for word in tokens:
        if word in w2v_model.wv:
            embeddings.append(w2v_model.wv[word])
        else:
            embeddings.append(np.zeros(embed_dim))
    if not embeddings:
        embeddings = [np.zeros(embed_dim)]
    avg_vector = np.mean(embeddings, axis=0)
    return np.reshape(avg_vector, (1, 1, embed_dim))


def explain_with_lime(text):
    def predict_proba(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=128)

        # Apply the same reshaping as in the predict function
        if padded_sequences.shape[1] < 100:
            padded_sequences = np.pad(padded_sequences,
                                      ((0, 0), (0, 100 -
                                       padded_sequences.shape[1])),
                                      mode='constant')
        elif padded_sequences.shape[1] > 100:
            padded_sequences = padded_sequences[:, :100]

        model_input = np.expand_dims(padded_sequences, axis=1)
        return model.predict(model_input)

    # Should match your model
    class_names = ["Negative", "Neutral", "Positive"]
    lime_explainer = LimeTextExplainer(class_names=class_names)
    explanation = lime_explainer.explain_instance(
        text, predict_proba, num_features=10)
    return {"words": explanation.as_list(), "html": explanation.as_html()}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tokens = preprocess_input(text)
    embedding = tweet_to_embedding(tokens)
    prediction = model.predict(embedding)[0]
    class_names = ["Negative", "Neutral", "Positive"]
    predicted_class = np.argmax(prediction)
    sentiment_label = class_names[predicted_class]
    confidence = float(prediction[predicted_class])
    sentiment = sentiment_pipeline(text)[0]
    sentiment_label = sentiment['label']

    explanation = explain_with_lime(text)

    return jsonify({
        "prediction": {
            "label": sentiment_label,
            "score": confidence,
            "scores": {name: float(score) for name, score in zip(class_names, prediction)},
            "text": text
        },
        "explanation": explanation
    })


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
