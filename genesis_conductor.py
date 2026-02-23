import runpy
import sys

# Shim to run simple_seismic_server.py as 'genesis-conductor'
# This satisfies the Render start command configuration.

if __name__ == "__main__":
    print("Genesis Conductor: Launching Seismic Log Server...")
    # Execute the server script
    runpy.run_path("simple_seismic_server.py", run_name="__main__")
