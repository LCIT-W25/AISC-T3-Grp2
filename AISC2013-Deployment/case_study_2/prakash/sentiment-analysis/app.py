from flask import Flask, render_template, request
from flask_cors import CORS
import numpy as np
import torch
from tensorflow.keras.models import load_model
from transformers import pipeline
import re
import emoji
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder="templates")
app.config['WTF_CSRF_ENABLED'] = True

# Load the RNN model (as .h5)
rnn_model = load_model('best_model_rnn.h5')

# Load the GRU model (as .pth)
# gru_model = torch.load('best_biGRU_model.pth')
# gru_model.eval()

# Sentiment analysis pipeline (for interpretability)
sentiment_pipeline = pipeline("sentiment-analysis")

# Preprocessing functions


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


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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


# Tokenizer setup (make sure it's the same tokenizer you used during training)
tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(df['Tweet'])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Preprocess the input text
    processed_text = preprocess_input(text)

    # Tokenization and Padding
    processed_seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(processed_seq, maxlen=50)

    # Get prediction from RNN model
    rnn_pred = rnn_model.predict(padded_seq)

    # Get prediction from GRU model
    # gru_input = torch.tensor(padded_seq)
    # gru_pred = gru_model(gru_input)

    # Interpretability (using HuggingFace Sentiment Pipeline)
    sentiment = sentiment_pipeline(text)[0]

    return render_template('index.html',
                           text=text,
                           rnn_result=rnn_pred,
                           #    gru_result=gru_pred.item(),
                           sentiment=sentiment['label'])


if __name__ == '__main__':
    app.run(debug=True)
