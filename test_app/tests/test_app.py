from app import app
import unittest

class FlaskTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
    
    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
