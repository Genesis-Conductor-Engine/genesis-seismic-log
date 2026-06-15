import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
import threading

from simple_seismic_server import SYSTEM_METRICS

# Create a sample data
DATA = {
    "timestamp": "2023-10-27T10:00:00.000000",
    "system": "GTX 1650 (Diamond Vault)",
    "metrics": SYSTEM_METRICS,
    "percentiles": {
        "p50": SYSTEM_METRICS["latency_p50_ms"],
        "p95": SYSTEM_METRICS["latency_p95_ms"],
        "p99": SYSTEM_METRICS["latency_p99_ms"],
        "p999": SYSTEM_METRICS["latency_p999_ms"]
    },
    "energy_efficiency": {
        "joules_per_op": SYSTEM_METRICS["energy_per_op_joules"],
        "comparison_cloud_joules_per_op": 100.0,
        "efficiency_gain": "2380x"
    },
    "verification": {
        "protocol": "S-ToT Seismic Stress",
        "status": SYSTEM_METRICS["crystallization_status"],
        "ground_truth": "Ed25519 attestation active"
    }
}

# Measure dumps time locally
N_ITERS = 100000

print("Testing locally...")
t0 = time.time()
for _ in range(N_ITERS):
    # Using indent=2 (what we want to replace if it exists, wait, it's currently using separators=(',', ':') based on my read of simple_seismic_server.py?)
    # Wait, my memory says "simple_seismic_server.py explicitly uses `json.dumps(data, separators=(',', ':'))` to produce compact, non-indented JSON responses for dynamic data."
    pass
