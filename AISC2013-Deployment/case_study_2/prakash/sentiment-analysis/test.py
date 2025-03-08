from app import app, preprocess_input, clean_text, handle_emojis
import unittest
from unittest.mock import patch, MagicMock
import nltk
import torch


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

    def test_clean_text_function(self):
        """Test the clean_text function specifically"""
        text = "Hello @username! Check out #hashtag and http://example.com"
        cleaned = clean_text(text)

        self.assertEqual(cleaned, "hello  check out  and ")

    def test_handle_emojis_function(self):
        """Test emoji handling function"""

        text = "I love this 😀 and that 👍"
        processed = handle_emojis(text)
        self.assertIn(":grinning_face:", processed)
        self.assertIn(":thumbs_up:", processed)

    @patch('app.gru_model')
    @patch('app.sentiment_pipeline')
    @patch('app.tokenizer')
    @patch('app.explainer.explain_instance')
    def test_predict_route(self, mock_explainer, mock_tokenizer, mock_sentiment_pipeline, mock_gru_model):
        """Test the prediction route with mocked models"""
        mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]

        mock_gru_pred = MagicMock()
        mock_gru_pred.dim.return_value = 1
        mock_argmax = MagicMock()
        mock_argmax.item.return_value = 2  # Positive class
        torch.argmax = MagicMock(return_value=mock_argmax)
        mock_gru_model.return_value = mock_gru_pred

        # Setup sentiment pipeline mock
        mock_sentiment_pipeline.return_value = [
            {'label': 'POSITIVE', 'score': 0.9}]

        # Setup LIME explainer mock
        mock_explanation = MagicMock()
        mock_explanation.as_html.return_value = "<div>Explanation HTML</div>"
        mock_explainer.return_value = mock_explanation

        # Test prediction
        response = self.app.post(
            '/predict', data={'text': 'This is a test message'})

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify our mocks were called
        mock_tokenizer.texts_to_sequences.assert_called()
        mock_sentiment_pipeline.assert_called_with('This is a test message')
        mock_explainer.assert_called()

    def test_empty_input(self):
        """Test behavior when empty text is submitted"""
        response = self.app.post('/predict', data={'text': ''})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please enter some text!', response.data)


if __name__ == '__main__':
    unittest.main()
