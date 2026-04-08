import time
import subprocess
import urllib.request
import urllib.error
import sys

SERVER_PORT = 8003
SERVER_URL = f"http://localhost:{SERVER_PORT}/api/bench/live"

def wait_for_server():
    for _ in range(20):
        try:
            with urllib.request.urlopen(SERVER_URL) as response:
                if response.status == 200:
                    return True
        except urllib.error.URLError:
            time.sleep(0.5)
        except Exception as e:
            print(f"Error checking server: {e}")
            time.sleep(0.5)
    return False

def benchmark(num_requests=100):
    start_time = time.time()
    total_bytes = 0

    for _ in range(num_requests):
        try:
            with urllib.request.urlopen(SERVER_URL) as response:
                content = response.read()
                total_bytes += len(content)
        except Exception as e:
            print(f"Request failed: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    avg_size = total_bytes / num_requests if num_requests > 0 else 0

    print(f"Total Requests: {num_requests}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Request Time: {(total_time / num_requests * 1000):.2f}ms")
    print(f"Average Response Size: {avg_size:.2f} bytes")

def main():
    print("Starting server...")
    server_process = subprocess.Popen([sys.executable, "simple_seismic_server.py"],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    try:
        if wait_for_server():
            print("Server is ready. Benchmarking...")
            benchmark(100)
        else:
            print("Server failed to start within timeout.")
    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
