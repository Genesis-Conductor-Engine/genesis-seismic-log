import urllib.request
import time
import threading

def run_bench():
    start = time.time()
    count = 0
    while time.time() - start < 5:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8003/") as response:
                response.read()
                count += 1
        except Exception as e:
            pass
    print(f"RPS: {count / 5}")

threads = []
for i in range(4):
    t = threading.Thread(target=run_bench)
    threads.append(t)
    t.start()
for t in threads:
    t.join()
