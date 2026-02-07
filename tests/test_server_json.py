import unittest
from unittest.mock import MagicMock
import json
import sys
import os

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_seismic_server import SeismicHandler

class TestSeismicHandler(SeismicHandler):
    def __init__(self):
        # Bypass BaseHTTPRequestHandler.__init__ to avoid socket operations
        self.wfile = MagicMock()
        self.headers = {}
        self.request_version = "HTTP/1.1"
        self.command = "GET"
        self.path = "/"

    def send_response(self, code, message=None):
        pass

    def send_header(self, keyword, value):
        pass

    def end_headers(self):
        pass

class TestServerJson(unittest.TestCase):
    def test_send_json_minified(self):
        """Verify that send_json produces minified JSON without indentation or spaces."""
        handler = TestSeismicHandler()
        data = {"foo": "bar", "baz": [1, 2], "nested": {"a": 1}}
        handler.send_json(data)

        # Get the arguments passed to wfile.write
        # call_args[0] is positional args, [0] is the first arg
        args = handler.wfile.write.call_args[0][0]
        output = args.decode('utf-8')

        # Check for absence of spaces and newlines
        self.assertNotIn('\n', output, "Output should not contain newlines")
        self.assertNotIn(' ', output, "Output should not contain spaces")

        # Verify exact format
        expected = '{"foo":"bar","baz":[1,2],"nested":{"a":1}}'
        self.assertEqual(output, expected)

if __name__ == '__main__':
    unittest.main()
