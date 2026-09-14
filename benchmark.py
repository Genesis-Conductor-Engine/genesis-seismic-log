import urllib.request
import time
import threading

def bench_endpoint(url, duration=2):
    count = 0
    start = time.time()
    while time.time() - start < duration:
        try:
            with urllib.request.urlopen(url) as response:
                response.read()
                count += 1
        except Exception:
            pass
    return count / duration

print("Starting server in background...")
import subprocess
p = subprocess.Popen(["python3", "simple_seismic_server.py"])
time.sleep(1)

print("Benchmarking / ...")
rps_root = bench_endpoint("http://localhost:8003/")
print(f"Root RPS: {rps_root}")

print("Benchmarking /api/health ...")
rps_health = bench_endpoint("http://localhost:8003/api/health")
print(f"Health RPS: {rps_health}")

p.terminate()
p.wait()
