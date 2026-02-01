import unittest
import threading
import time
import urllib.request
import json
from http.server import HTTPServer
import sys
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_seismic_server import SeismicHandler

class TestSeismicServerLive(unittest.TestCase):
    def setUp(self):
        # Use port 0 to let OS choose a free port
        self.server = HTTPServer(('localhost', 0), SeismicHandler)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(0.1) # Give it time to start

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_json_is_minified(self):
        url = f"http://localhost:{self.port}/api/bench/live"
        with urllib.request.urlopen(url) as response:
            content = response.read()
            text = content.decode('utf-8')

            # Check for absence of formatting characters
            self.assertNotIn('\n', text, "Response should not contain newlines")

            # Spaces are tricky because timestamps or text might have spaces.
            # But separators=(',', ':') removes spaces after comma and colon.
            # So ": " should not exist, ", " should not exist.
            # However, we must be careful if the content *values* contain ": " or ", ".
            # The keys are simple strings. The values might have them.
            # Let's check a specific snippet we know shouldn't have spaces, like between keys.

            self.assertNotIn('": ', text, "Should not have space after colon in key-value pairs")
            self.assertNotIn(', "', text, "Should not have space after comma between fields")

            # Verify it is valid JSON
            data = json.loads(text)
            self.assertIn("metrics", data)

            # Also check the / endpoint
            url_root = f"http://localhost:{self.port}/"
            with urllib.request.urlopen(url_root) as response_root:
                text_root = response_root.read().decode('utf-8')
                self.assertNotIn('\n', text_root)
                self.assertNotIn('": ', text_root)

if __name__ == '__main__':
    unittest.main()
