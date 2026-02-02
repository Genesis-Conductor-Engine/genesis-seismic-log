
import unittest
import json
import io
import sys
import os

# Add parent directory to path to import simple_seismic_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_seismic_server import SeismicHandler

class MockRequest:
    def makefile(self, *args, **kwargs):
        return io.BytesIO()

class MockServer:
    pass

class MockWFile(io.BytesIO):
    def close(self):
        pass

class TestSeismicServerPerf(unittest.TestCase):
    def test_send_json_minification(self):
        """
        Verify that send_json produces minified JSON without indentation or extra spaces.
        """
        # Mocking the handler setup
        request = MockRequest()
        client_address = ('127.0.0.1', 8888)
        server = MockServer()

        # We can't instantiate SeismicHandler easily because it expects a socket.
        # So we mock the methods we need or subclass it to avoid __init__ issues,
        # OR we just instantiate it with a mock request.

        # BaseHTTPRequestHandler calls setup() in __init__.
        # Let's try to mock the necessary parts.

        handler = SeismicHandler(request, client_address, server)

        # Manually set attributes that are usually set during request handling
        handler.requestline = "GET /TEST HTTP/1.1"
        handler.request_version = "HTTP/1.1"

        # Override wfile with our capture buffer
        handler.wfile = MockWFile()

        test_data = {
            "foo": "bar",
            "numbers": [1, 2, 3],
            "nested": {"a": 1}
        }

        # We need to suppress the actual socket writes (send_response etc)
        # send_response writes to wfile as well.

        # Let's intercept the final write which contains the body.
        # But send_json calls send_response...

        handler.send_json(test_data)

        output = handler.wfile.getvalue()

        # The output includes headers. We need to split body.
        # Headers end with \r\n\r\n

        headers, body_bytes = output.split(b'\r\n\r\n', 1)
        body_str = body_bytes.decode('utf-8')

        # Assertions
        self.assertNotIn('\n', body_str, "JSON body should not contain newlines")
        self.assertNotIn(' ', body_str, "JSON body should not contain spaces")

        expected = '{"foo":"bar","numbers":[1,2,3],"nested":{"a":1}}'
        self.assertEqual(body_str, expected)

if __name__ == '__main__':
    unittest.main()
