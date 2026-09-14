import urllib.request
import time
import threading

def worker(url, duration, count_list, index):
    start = time.time()
    count = 0
    while time.time() - start < duration:
        try:
            with urllib.request.urlopen(url) as response:
                response.read()
                count += 1
        except:
            pass
    count_list[index] = count

def run_benchmark(url, duration=5, threads=4):
    counts = [0] * threads
    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=worker, args=(url, duration, counts, i))
        thread_list.append(t)
        t.start()

    for t in thread_list:
        t.join()

    total_requests = sum(counts)
    rps = total_requests / duration
    print(f"URL: {url}")
    print(f"Total Requests: {total_requests}")
    print(f"RPS: {rps:.2f}")
    return rps

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8003/"
    run_benchmark(url)
