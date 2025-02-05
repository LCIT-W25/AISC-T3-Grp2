import pytest
from flask import Flask
from app2 import app  # Ensure your Flask app is named 'app.py'
import json

def test_homepage():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data  # Ensure it returns HTML content

def test_us_page():
    client = app.test_client()
    response = client.get('/Us')
    assert response.status_code == 200
    assert b"<!DOCTYPE html" in response.data

def test_predict_valid_json():
    client = app.test_client()
    payload = json.dumps({"text": "I love this product!", "model": "NB"})
    response = client.post('/predict', data=payload, content_type='application/json')
    assert response.status_code == 200
    assert "prediction" in response.get_json()

def test_predict_empty_text():
    client = app.test_client()
    payload = json.dumps({"text": "", "model": "NB"})
    response = client.post('/predict', data=payload, content_type='application/json')
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_predict_invalid_request():
    client = app.test_client()
    response = client.post('/predict', data="notjson", content_type='text/plain')
    assert response.status_code == 400
    assert "error" in response.get_json()

if __name__ == "__main__":
    pytest.main()