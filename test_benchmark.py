import urllib.request
import time
import threading

URL = "http://localhost:8003/"
NUM_REQUESTS = 1000

def make_requests():
    for _ in range(NUM_REQUESTS):
        try:
            urllib.request.urlopen(URL).read()
        except Exception:
            pass

def run_benchmark():
    threads = []
    start_time = time.time()

    for _ in range(10):
        t = threading.Thread(target=make_requests)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    duration = end_time - start_time
    total_requests = NUM_REQUESTS * 10
    rps = total_requests / duration

    print(f"Total Requests: {total_requests}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Requests per Second: {rps:.2f}")

if __name__ == "__main__":
    run_benchmark()
