import os
import re
import numpy as np
import emoji
import contractions
import nltk
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
from tensorflow.keras import layers
from tensorflow import keras


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])


class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embedding_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        return self.position_embeddings(positions)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)  # Residual Connection

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


# Load the trained Keras model with the custom layer
model_path = "non_causal_tf.keras"
with custom_object_scope({'TransformerBlock': TransformerBlock, 'PositionEmbedding': PositionEmbedding}):
    model = load_model(model_path)

print(model.layers[-1].get_config())


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing functions (same as before)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def handle_emojis(text):
    return emoji.demojize(text)


def expand_contractions(text):
    return contractions.fix(text)


slang_dict = {"u": "you", "ur": "your", "idk": "i don't know",
              "btw": "by the way", "smh": "shaking my head"}


def replace_slang(text):
    words = text.split()
    return ' '.join([slang_dict[word] if word in slang_dict else word for word in words])


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def preprocess_input(text):
    text = clean_text(text)
    text = handle_emojis(text)
    text = expand_contractions(text)
    text = replace_slang(text)
    text = preprocess_text(text)
    return text


# Load tokenizer and other components
tokenizer = Tokenizer(num_words=5000)

# Initialize LIME explainer (same as before)
lime_explainer = LimeTextExplainer(class_names=["negative", "positive"])

# Flask API setup
app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess the text
    processed_text = preprocess_input(text)

    print("Processed text:", processed_text)

    # Tokenize and pad the sequence
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=128)

    # Reshape the input to match model expectations (None, 1, 100)
    if padded_sequences.shape[1] < 100:
        padded_sequences = np.pad(padded_sequences,
                                  ((0, 0), (0, 100 -
                                   padded_sequences.shape[1])),
                                  mode='constant')
    elif padded_sequences.shape[1] > 100:
        padded_sequences = padded_sequences[:, :100]

    # Add the extra dimension
    model_input = np.expand_dims(padded_sequences, axis=1)

    # Make prediction
    prediction = model.predict(model_input)[0]  # Get first batch item

    # Determine class with highest probability
    # Adjust based on your model
    class_names = ["negative", "neutral", "positive"]
    predicted_class = np.argmax(prediction)
    sentiment_label = class_names[predicted_class]
    confidence = float(prediction[predicted_class])

    # Generate explanation
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
    class_names = ["negative", "neutral", "positive"]
    lime_explainer = LimeTextExplainer(class_names=class_names)
    explanation = lime_explainer.explain_instance(
        text, predict_proba, num_features=10)
    return {"words": explanation.as_list(), "html": explanation.as_html()}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
