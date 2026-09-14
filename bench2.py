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
    return total / duration

if __name__ == "__main__":
    server = run_server()
    time.sleep(1) # wait for server to start

    print("Baseline Benchmarking:")
    rps_root = run_bench("http://127.0.0.1:8003/", duration=4, threads=4)
    print(f"RPS /: {rps_root:.2f}")

    rps_health = run_bench("http://127.0.0.1:8003/api/health", duration=4, threads=4)
    print(f"RPS /api/health: {rps_health:.2f}")

    server.terminate()
