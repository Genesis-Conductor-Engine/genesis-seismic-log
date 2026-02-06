
import unittest
import json
from unittest.mock import Mock
from simple_seismic_server import SeismicHandler

class TestSeismicHandler(SeismicHandler):
    def __init__(self):
        pass

class TestSeismicJSON(unittest.TestCase):
    def test_send_json_minification(self):
        """Verify that send_json sends minified JSON without whitespace"""

        # Instantiate our test handler which skips __init__
        handler = TestSeismicHandler()

        # Mock wfile to capture output
        handler.wfile = Mock()
        handler.wfile.write = Mock()

        # Mock headers methods as they are from BaseHTTPRequestHandler
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        test_data = {"key": "value", "list": [1, 2, 3]}

        # Call the method we want to test
        handler.send_json(test_data)

        # Get the argument passed to wfile.write
        args, _ = handler.wfile.write.call_args
        output_bytes = args[0]
        output_str = output_bytes.decode('utf-8')

        # Check for minification (no spaces after comma or colon)
        self.assertNotIn(": ", output_str, "Output should not contain space after colon")
        self.assertNotIn(", ", output_str, "Output should not contain space after comma")
        self.assertNotIn("\n", output_str, "Output should not contain newlines")

        # Verify correctness
        self.assertEqual(json.loads(output_str), test_data)

if __name__ == '__main__':
    unittest.main()
