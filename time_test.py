import urllib.request
import time
import threading
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = { "a": 1, "b": "hello" }
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    def log_message(self, format, *args):
        pass

class MockHandlerCompact(BaseHTTPRequestHandler):
    def do_GET(self):
        data = { "a": 1, "b": "hello" }
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, separators=(',', ':')).encode())
    def log_message(self, format, *args):
        pass

PORT1 = 8031
server1 = ThreadingHTTPServer(('127.0.0.1', PORT1), MockHandler)
threading.Thread(target=server1.serve_forever, daemon=True).start()

PORT2 = 8032
server2 = ThreadingHTTPServer(('127.0.0.1', PORT2), MockHandlerCompact)
threading.Thread(target=server2.serve_forever, daemon=True).start()

time.sleep(0.5)

def run(port, iters=5000):
    start = time.time()
    for _ in range(iters):
        req = urllib.request.Request(f'http://127.0.0.1:{port}/')
        with urllib.request.urlopen(req) as response:
            response.read()
    return time.time() - start

t1 = run(PORT1)
t2 = run(PORT2)
print("indent", t1)
print("compact", t2)
