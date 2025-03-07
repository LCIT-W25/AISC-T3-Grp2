import os
import io
import json
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from app import app, preprocess_image, UPLOAD_FOLDER, EXPLANATION_FOLDER, predict_proba_for_lime


class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        """Set up test client and other test variables."""
        self.app = app.test_client()
        self.app.testing = True

        # Ensure test directories exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(EXPLANATION_FOLDER, exist_ok=True)

        test_image_path = "burger.jpg"

        # create mock image if the file doesn't exist
        if os.path.exists(test_image_path):
            self.test_image = Image.open(test_image_path).convert("RGB")
        else:
            print(
                f"Warning: Test image not found at {test_image_path}.")
            self.test_image = Image.new('RGB', (200, 200), color='blue')
            self.test_image.save("generated_test_image.jpg")

        # Convert the image to a file-like object
        self.test_image_io = io.BytesIO()
        self.test_image.save(self.test_image_io, format='JPEG')
        self.test_image_io.seek(0)

        # Class 2 has highest probability
        self.mock_predictions = np.array([[0.1, 0.2, 0.5, 0.1, 0.1]])

    def test_home_page(self):
        """Test that home page is accessible."""
        response = self.app.get('/')

        self.assertEqual(response.status_code, 200)  # expected status code 200

    def test_preprocess_image_cnn(self):
        """Test image preprocessing for CNN model."""
        processed = preprocess_image(self.test_image, "CNN")

        self.assertEqual(processed.shape, (1, 128, 128, 3))
        self.assertTrue(0 <= processed.max() <= 1)

    def test_preprocess_image_efficientnet(self):
        """Test image preprocessing for EfficientNet model."""
        processed = preprocess_image(self.test_image, "EfficientNet")

        self.assertEqual(processed.shape, (1, 224, 224, 3))
        self.assertTrue(0 <= processed.max() <= 1)

    def test_predict_proba_for_lime_cnn(self):
        """Test the LIME prediction function for CNN model."""
        # mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = self.mock_predictions

        # test images
        test_images = np.array([np.array(self.test_image.resize((128, 128)))])

        result = predict_proba_for_lime(test_images, mock_model, "CNN")

        mock_model.predict.assert_called_once()
        self.assertEqual(result.shape, (1, 5))

    def test_predict_proba_for_lime_efficientnet(self):
        """Test the LIME prediction function for EfficientNet model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = self.mock_predictions
        test_images = np.array([np.array(self.test_image.resize((224, 224)))])
        result = predict_proba_for_lime(
            test_images, mock_model, "EfficientNet")

        # check if model was called and returned the expected result
        mock_model.predict.assert_called_once()
        self.assertEqual(result.shape, (1, 5))

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
            'file': (self.test_image_io, 'burger.jpg')
        })
        print(response.data)
        print(json.loads(response.data))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.data)[
                         'error'], 'No model selected')  # expected error message

    @patch('app.lime_image.LimeImageExplainer')
    @patch('app.cnn_model')
    @patch('matplotlib.pyplot')
    def test_predict_endpoint_success(self, mock_plt, mock_model, mock_lime):
        """Test successful prediction."""

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

    @patch('app.lime_image.LimeImageExplainer')
    @patch('app.efficientnet_model')
    @patch('matplotlib.pyplot')
    def test_predict_endpoint_efficientnet_success(self, mock_plt, mock_model, mock_lime):
        """Test successful prediction with EfficientNet model."""
        # Mock the model prediction
        mock_model.predict.return_value = self.mock_predictions

        # Mock LIME explainer
        mock_explainer = MagicMock()
        mock_explanation = MagicMock()
        # The predicted class (index 2) is available
        mock_explanation.top_labels = [2]
        mock_explanation.get_image_and_mask.return_value = (
            np.zeros((128, 128, 3)), np.zeros((128, 128)))
        mock_explainer.explain_instance.return_value = mock_explanation
        mock_lime.return_value = mock_explainer

        response = self.app.post('/predict', data={
            'model': 'EfficientNet',
            'file': (self.test_image_io, 'burger.jpg')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        # Class with highest probability
        self.assertEqual(data['prediction'], 'inside')
        self.assertEqual(data['model_used'], 'EfficientNet')
        self.assertIn('confidence', data)

    @patch('app.cnn_model')
    def test_predict_endpoint_lime_failure(self, mock_model):
        """Test that prediction works even if LIME explanation fails."""
        # Mock the model prediction
        mock_model.predict.return_value = self.mock_predictions

        # Patch the LIME explainer to raise an exception
        with patch('app.lime_image.LimeImageExplainer', side_effect=Exception("LIME failed")):
            # Make the request
            response = self.app.post('/predict', data={
                'model': 'CNN',
                'file': (self.test_image_io, 'burger.jpg')
            }, content_type='multipart/form-data')

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['prediction'], 'inside')
            self.assertEqual(data['model_used'], 'CNN')
            self.assertIn('confidence', data)


if __name__ == '__main__':
    unittest.main()
