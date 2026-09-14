import urllib.request
import time

def benchmark():
    start = time.time()
    for _ in range(5000):
        try:
            with urllib.request.urlopen('http://localhost:8003/') as response:
                response.read()
        except Exception as e:
            print("Error:", e)
            break
    end = time.time()
    print(f"Root endpoint - Total time for 5000 requests: {end - start:.4f} seconds")
    print(f"Root endpoint - Requests per second: {5000 / (end - start):.2f}")

if __name__ == "__main__":
    benchmark()
