import unittest
from unittest.mock import patch

# Patch pickle before importing app2 to prevent unpickling errors
with patch("pickle.load", lambda x: None):
    from app2 import app

class TestAppBasic(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_homepage(self):
        """Test if homepage loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html', response.data)

    # def test_us_page(self):
    #     """Test if Us page loads correctly."""
    #     response = self.client.get('/Us')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn(b'<!DOCTYPE html', response.data)

if __name__ == "__main__":
    unittest.main()
