import unittest
import os
import json
import io
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from app import app, preprocess_image, UPLOAD_FOLDER, EXPLANATION_FOLDER


class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        """Set up test client and other test variables."""
        self.app = app.test_client()
        self.app.testing = True

        # Ensure test directories exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(EXPLANATION_FOLDER, exist_ok=True)

        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_io = io.BytesIO()
        self.test_image.save(self.test_image_io, format='JPEG')
        self.test_image_io.seek(0)

        # Mock predictions
        # Class 2 has highest probability
        self.mock_predictions = np.array([[0.1, 0.2, 0.5, 0.1, 0.1]])

    def test_home_page(self):
        """Test that home page is accessible."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_preprocess_image(self):
        """Test image preprocessing for both models."""
        # Test CNN preprocessing
        cnn_processed = preprocess_image(self.test_image, "CNN")
        self.assertEqual(cnn_processed.shape, (1, 128, 128, 3))

        # Test EfficientNet preprocessing
        efficientnet_processed = preprocess_image(
            self.test_image, "EfficientNet")
        self.assertEqual(efficientnet_processed.shape, (1, 224, 224, 3))

    @patch('app.cnn_model')
    def test_predict_endpoint_errors(self, mock_model):
        """Test error handling in predict endpoint."""
        # Test no file uploaded
        response = self.app.post('/predict', data={'model': 'CNN'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.data)[
                         'error'], 'No file uploaded')

        # Test no model selected
        response = self.app.post('/predict', data={
            'file': (self.test_image_io, 'test_image.jpg')
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.data)[
                         'error'], 'No model selected')

    @patch('app.lime_image.LimeImageExplainer')
    @patch('app.cnn_model')
    @patch('matplotlib.pyplot')
    def test_predict_endpoint_success(self, mock_plt, mock_model, mock_lime):
        """Test successful prediction."""
        # Mock the model prediction
        mock_model.predict.return_value = self.mock_predictions

        # Mock LIME explainer
        mock_explainer = MagicMock()
        mock_explanation = MagicMock()
        mock_explanation.top_labels = [2]
        mock_explanation.get_image_and_mask.return_value = (
            np.zeros((128, 128, 3)), np.zeros((128, 128)))
        mock_explainer.explain_instance.return_value = mock_explanation
        mock_lime.return_value = mock_explainer

        # Make the request
        response = self.app.post('/predict', data={
            'model': 'CNN',
            'file': (self.test_image_io, 'burger.jpg')
        }, content_type='multipart/form-data')

        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['prediction'], 'inside')
        self.assertEqual(data['model_used'], 'CNN')


if __name__ == '__main__':
    unittest.main()
