from app import app, preprocess_input
import unittest
from unittest.mock import patch, MagicMock
import json
import numpy as np
import torch
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        """Test that the home route returns the index page"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        # Assuming your template has this
        self.assertIn(b'<!DOCTYPE html>', response.data)

    # @patch('app.rnn_model')
    # @patch('app.gru_model')
    # @patch('app.sentiment_pipeline')
    # @patch('app.tokenizer')
    # def test_predict_route(self, mock_tokenizer, mock_sentiment_pipeline, mock_gru_model, mock_rnn_model):
    #     """Test the prediction route with mocked models"""
    #     # Setup mocks
    #     mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    #     mock_rnn_model.predict.return_value = np.array([[0.7]])

    #     mock_gru_output = MagicMock()
    #     mock_gru_output.item.return_value = 0.8
    #     mock_gru_model.return_value = mock_gru_output

    #     mock_sentiment_pipeline.return_value = [
    #         {'label': 'POSITIVE', 'score': 0.9}]

    #     # Test prediction
    #     response = self.app.post(
    #         '/predict', data={'text': 'This is a test message'})

    #     # Verify response
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn(b'This is a test message', response.data)

    #     # Verify our mocks were called
    #     mock_tokenizer.texts_to_sequences.assert_called_once()
    #     mock_rnn_model.predict.assert_called_once()
    #     mock_sentiment_pipeline.assert_called_once_with(
    #         'This is a test message')

    def test_preprocess_input(self):
        """Test the text preprocessing function"""
        # Test with a sample text containing various elements to clean
        sample_text = "Hey @user, check out this link http://example.com #example I'm smh at ur idk moment 😂"
        processed = preprocess_input(sample_text)

        print(processed)

        self.assertNotIn('@user', processed)
        self.assertNotIn('http', processed)
        self.assertNotIn('#example', processed)

        self.assertNotIn('ur', processed)
        self.assertNotIn('idk', processed)
        self.assertNotIn('smh', processed)

        self.assertNotIn('😂', processed)

        self.assertNotIn("I'm", processed.lower())

        self.assertEqual(processed, processed.lower())  # expect all lowercase


if __name__ == '__main__':
    unittest.main()
