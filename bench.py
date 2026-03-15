import urllib.request
import time
import threading

def bench_endpoint(url, duration=5):
    count = 0
    start = time.time()
    end = start + duration
    while time.time() < end:
        try:
            with urllib.request.urlopen(url) as response:
                response.read()
            count += 1
        except Exception:
            pass
    return count / duration

if __name__ == "__main__":
    print("Starting server in background...")
    import subprocess
    server = subprocess.Popen(["python3", "simple_seismic_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1) # wait for server to start
    try:
        rps = bench_endpoint("http://localhost:8003/")
        print(f"Baseline RPS for / : {rps:.2f}")
    finally:
        server.terminate()
        server.wait()
