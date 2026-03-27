import urllib.request
import time
import threading

def worker(url, num_requests, results, index):
    start = time.time()
    success = 0
    for _ in range(num_requests):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                response.read()
                success += 1
        except Exception:
            pass
    results[index] = {"time": time.time() - start, "success": success}

def bench(url, total_requests, num_threads):
    print(f"Benchmarking {url} with {total_requests} requests over {num_threads} threads...")
    reqs_per_thread = total_requests // num_threads
    results = [None] * num_threads
    threads = []

    start_time = time.time()
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(url, reqs_per_thread, results, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    total_time = end_time - start_time
    total_success = sum(r["success"] for r in results)

    print(f"Total time: {total_time:.4f}s")
    print(f"Successful requests: {total_success}/{total_requests}")
    print(f"Requests per second: {total_success / total_time:.2f}")

if __name__ == '__main__':
    bench("http://localhost:8003/api/bench/live", 5000, 10)
