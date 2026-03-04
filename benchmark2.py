import time
import json
from simple_seismic_server import SYSTEM_METRICS

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

N = 100000

t0 = time.time()
for _ in range(N):
    json.dumps(DATA, indent=2)
t1 = time.time()
print(f"indent=2: {t1 - t0:.4f} seconds")

t0 = time.time()
for _ in range(N):
    json.dumps(DATA, separators=(',', ':'))
t1 = time.time()
print(f"separators=(',', ':'): {t1 - t0:.4f} seconds")

t0 = time.time()
for _ in range(N):
    json.dumps(DATA)
t1 = time.time()
print(f"default: {t1 - t0:.4f} seconds")
