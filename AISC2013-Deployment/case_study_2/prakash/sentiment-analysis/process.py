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


text = preprocess_input(
    "Hey @user, check out this link http://example.com #example I'm smh at ur idk moment 😂")

print(text)
