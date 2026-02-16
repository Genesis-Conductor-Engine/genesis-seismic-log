import time
import subprocess
import urllib.request
import urllib.error
import sys
import json

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

def verify_json():
    try:
        with urllib.request.urlopen(SERVER_URL) as response:
            content = response.read()
            text = content.decode('utf-8')

            # Verify JSON validity
            try:
                data = json.loads(text)
                print("✅ Response is valid JSON.")
            except json.JSONDecodeError:
                print("❌ Response is NOT valid JSON.")
                sys.exit(1)

            # Verify compactness
            if '\n' in text:
                print("❌ Response contains newlines (indentation detected).")
                sys.exit(1)

            if ': ' in text:
                print("❌ Response contains spaces after colons (default separators detected).")
                sys.exit(1)

            if ', ' in text:
                print("❌ Response contains spaces after commas (default separators detected).")
                sys.exit(1)

            print("✅ Response is compact (no indentation, minimal separators).")
            print(f"Payload size: {len(content)} bytes")

    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)

def main():
    print("Starting server for verification...")
    server_process = subprocess.Popen([sys.executable, "simple_seismic_server.py"],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    try:
        if wait_for_server():
            print("Server is ready.")
            verify_json()
        else:
            print("Server failed to start within timeout.")
            sys.exit(1)
    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
