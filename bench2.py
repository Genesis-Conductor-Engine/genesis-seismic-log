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
        except Exception as e:
            print(f"Error: {e}")
    return count / duration

if __name__ == "__main__":
    print("Starting server in background...")
    import subprocess
    server = subprocess.Popen(["python3", "simple_seismic_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1) # wait for server to start
    try:
        rps_root = bench_endpoint("http://localhost:8003/")
        print(f"Optimized RPS for / : {rps_root:.2f}")
        rps_health = bench_endpoint("http://localhost:8003/api/health")
        print(f"Optimized RPS for /api/health : {rps_health:.2f}")
    finally:
        server.terminate()
        server.wait()
