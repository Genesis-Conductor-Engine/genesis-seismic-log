import urllib.request
import time
import threading

def worker(url, duration, results):
    start = time.time()
    count = 0
    while time.time() - start < duration:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                response.read()
            count += 1
        except Exception:
            pass
    results.append(count)

def run_benchmark(url, duration=5, num_threads=10):
    threads = []
    results = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(url, duration, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_requests = sum(results)
    rps = total_requests / duration
    print(f"URL: {url}")
    print(f"Total requests: {total_requests}")
    print(f"Duration: {duration}s")
    print(f"RPS: {rps:.2f}")

if __name__ == "__main__":
    run_benchmark("http://127.0.0.1:8003/")
