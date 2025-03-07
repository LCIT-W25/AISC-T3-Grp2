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
