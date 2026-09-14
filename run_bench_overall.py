import urllib.request
import time
import threading
import json
import logging
from simple_seismic_server import SeismicHandler, SYSTEM_METRICS
from http.server import ThreadingHTTPServer
import sys
import os

# Suppress logging
class NoLogHandler(SeismicHandler):
    def log_message(self, format, *args):
        pass

class IndentHandler(NoLogHandler):
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

class CompactHandler(NoLogHandler):
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, separators=(',', ':')).encode())

def start_server(port, handler_class):
    server = ThreadingHTTPServer(('127.0.0.1', port), handler_class)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def benchmark(port, name, reqs=10000):
    start = time.time()
    for _ in range(reqs):
        req = urllib.request.Request(f'http://127.0.0.1:{port}/api/bench/live')
        try:
            with urllib.request.urlopen(req) as response:
                response.read()
        except Exception as e:
            pass
    end = time.time()
    print(f"[{name}] Time for {reqs} requests: {end - start:.2f} seconds")
    return end - start

s1 = start_server(8021, IndentHandler)
s2 = start_server(8022, CompactHandler)
time.sleep(0.5)

t1 = benchmark(8021, "indent=2")
t2 = benchmark(8022, "compact")

print(f"Speedup: {((t1-t2)/t1)*100:.2f}%")
