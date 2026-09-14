import urllib.request
import time
import threading
import json
from simple_seismic_server import SeismicHandler, SYSTEM_METRICS
from http.server import ThreadingHTTPServer

class TestHandler(SeismicHandler):
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

PORT = 8011
server = ThreadingHTTPServer(('127.0.0.1', PORT), TestHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.5)

def benchmark(name):
    start = time.time()
    for _ in range(5000):
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/api/bench/live')
        try:
            with urllib.request.urlopen(req) as response:
                response.read()
        except Exception:
            pass
    end = time.time()
    print(f"[{name}] Time for 5000 requests: {end - start:.2f} seconds")

benchmark("indent=2")
