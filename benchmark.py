import urllib.request
import time
import threading
import json

def fetch_root(count):
    for _ in range(count):
        try:
            req = urllib.request.Request("http://localhost:8003/")
            with urllib.request.urlopen(req) as response:
                response.read()
        except Exception:
            pass

def run_benchmark(concurrency, total_requests):
    start_time = time.time()
    threads = []
    requests_per_thread = total_requests // concurrency

    for _ in range(concurrency):
        t = threading.Thread(target=fetch_root, args=(requests_per_thread,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    duration = time.time() - start_time
    rps = total_requests / duration
    print(f"Concurrency: {concurrency}, Total Requests: {total_requests}")
    print(f"Time: {duration:.2f}s, RPS: {rps:.2f}")

if __name__ == "__main__":
    run_benchmark(10, 5000)
