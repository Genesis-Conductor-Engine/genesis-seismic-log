import urllib.request
import time
import threading

def benchmark(url, duration=5, threads=4):
    stop_event = threading.Event()
    counts = [0] * threads

    def run(thread_id):
        while not stop_event.is_set():
            try:
                with urllib.request.urlopen(url) as response:
                    response.read()
                    counts[thread_id] += 1
            except Exception:
                pass

    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=run, args=(i,))
        thread_list.append(t)
        t.start()

    time.sleep(duration)
    stop_event.set()

    for t in thread_list:
        t.join()

    total_requests = sum(counts)
    rps = total_requests / duration
    print(f"URL: {url}")
    print(f"Duration: {duration}s")
    print(f"Total Requests: {total_requests}")
    print(f"RPS: {rps:.2f}")

if __name__ == "__main__":
    benchmark("http://127.0.0.1:8003/")
