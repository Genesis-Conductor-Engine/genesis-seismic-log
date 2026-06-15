import urllib.request
import time
import threading

URL = "http://localhost:8003/"
NUM_REQUESTS = 5000
CONCURRENCY = 10

def worker(num_reqs):
    for _ in range(num_reqs):
        try:
            with urllib.request.urlopen(URL) as response:
                response.read()
        except Exception as e:
            pass

def run_benchmark():
    threads = []
    reqs_per_thread = NUM_REQUESTS // CONCURRENCY

    start_time = time.time()

    for _ in range(CONCURRENCY):
        t = threading.Thread(target=worker, args=(reqs_per_thread,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    elapsed = end_time - start_time
    rps = NUM_REQUESTS / elapsed

    print(f"Completed {NUM_REQUESTS} requests in {elapsed:.2f} seconds.")
    print(f"RPS: {rps:.2f}")

if __name__ == "__main__":
    run_benchmark()
