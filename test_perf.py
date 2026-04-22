import urllib.request
import time
import threading
from simple_seismic_server import SeismicHandler, SYSTEM_METRICS
from http.server import ThreadingHTTPServer

PORT = 8011
server = ThreadingHTTPServer(('127.0.0.1', PORT), SeismicHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.5)

def benchmark():
    start = time.time()
    for _ in range(10000):
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/api/bench/live')
        try:
            with urllib.request.urlopen(req) as response:
                pass
        except Exception:
            pass
    end = time.time()
    print(f"Time for 10000 requests: {end - start:.2f} seconds")

benchmark()
