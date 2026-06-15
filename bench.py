import urllib.request
import threading
import time
import subprocess

def run_server():
    return subprocess.Popen(["python3", "simple_seismic_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def worker(url, duration, results):
    start = time.time()
    count = 0
    while time.time() - start < duration:
        try:
            urllib.request.urlopen(url).read()
            count += 1
        except Exception:
            pass
    results.append(count)

def run_bench(url, duration=5, threads=4):
    results = []
    threads_list = []
    for _ in range(threads):
        t = threading.Thread(target=worker, args=(url, duration, results))
        t.start()
        threads_list.append(t)
    for t in threads_list:
        t.join()

    total = sum(results)
    print(f"URL: {url}")
    print(f"Total Requests: {total}")
    print(f"RPS: {total / duration:.2f}")
    return total / duration

if __name__ == "__main__":
    server = run_server()
    time.sleep(1) # wait for server to start
    print("Benchmarking / ...")
    rps_root = run_bench("http://127.0.0.1:8003/")
    print("Benchmarking /api/health ...")
    rps_health = run_bench("http://127.0.0.1:8003/api/health")
    server.terminate()
