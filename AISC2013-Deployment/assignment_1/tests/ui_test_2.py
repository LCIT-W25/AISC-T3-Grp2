import os
import unittest
from app import app
from io import BytesIO
from werkzeug.datastructures import FileStorage


class FlaskAppTestCase(unittest.TestCase):

    # Set up the test client
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test prediction endpoint for kNN model
    def test_knn_prediction(self):
        # Open a test image file (replace with an actual image in your project)
        # Change this to an actual image file
        image_path = 'app/static/uploads/test.jpg'
        with open(image_path, 'rb') as img:
            data = {
                'file': (BytesIO(img.read()), os.path.basename(image_path))
            }
            # Send POST request with image to the prediction route
            response = self.app.post(
                '/predict/knn', data=data, content_type='multipart/form-data')

        # Check if the response is 200 OK
        self.assertEqual(response.status_code, 200)

        # Check if prediction text is in the response
        self.assertIn('Prediction:', response.data.decode())

    # Test prediction endpoint for DNN model
    def test_dnn_prediction(self):
        # Change this to an actual image file
        image_path = 'app/static/uploads/test.jpg'
        with open(image_path, 'rb') as img:
            data = {
                'file': (BytesIO(img.read()), os.path.basename(image_path))
            }
            response = self.app.post(
                '/predict/dnn', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        self.assertIn('Prediction:', response.data.decode())

    # Test if no file was uploaded
    def test_no_file_uploaded(self):
        response = self.app.post(
            '/predict/knn', data={}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn('No file part', response.data.decode())


if __name__ == '__main__':
    unittest.main()
