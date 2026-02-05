
import unittest
from unittest.mock import MagicMock
import sys
import os
import json
from io import BytesIO

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_seismic_server import SeismicHandler

class TestSeismicHandler(unittest.TestCase):
    def test_send_json_minification(self):
        # Setup
        mock_wfile = BytesIO()

        # Create instance without invoking __init__ which triggers request handling
        handler = SeismicHandler.__new__(SeismicHandler)
        handler.wfile = mock_wfile
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        data = {"key": "value", "list": [1, 2, 3]}

        # Execute
        handler.send_json(data)

        # Verify
        content = mock_wfile.getvalue().decode('utf-8')
        # Minified JSON should look like this
        expected = '{"key":"value","list":[1,2,3]}'

        self.assertEqual(content, expected)
        self.assertNotIn(" ", content)
        self.assertNotIn("\n", content)

if __name__ == '__main__':
    unittest.main()
