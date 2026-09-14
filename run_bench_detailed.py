import time
import json
from datetime import datetime

SYSTEM_METRICS = {
    "hash_throughput_ops_sec": 15265,
    "latency_p50_ms": 1.1,
    "latency_p95_ms": 1.8,
    "latency_p99_ms": 2.0,
    "latency_p999_ms": 3.2,
    "energy_per_op_joules": 0.042,
    "gpu_model": "GTX 1650",
    "speedup_vs_cloud": "200x+",
    "crystallization_status": "CRYSTALLINE"
}

data = {
    "timestamp": datetime.utcnow().isoformat(),
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

def bench(name, serialize_func, iterations=100000):
    start = time.time()
    for _ in range(iterations):
        serialize_func(data)
    end = time.time()
    print(f"[{name}] {end - start:.4f}s")
    return end - start

t1 = bench("indent=2", lambda d: json.dumps(d, indent=2).encode())
t2 = bench("compact", lambda d: json.dumps(d, separators=(',', ':')).encode())

print(f"Speedup: {((t1-t2)/t1)*100:.2f}%")

print(f"Size indent=2: {len(json.dumps(data, indent=2).encode())} bytes")
print(f"Size compact: {len(json.dumps(data, separators=(',', ':')).encode())} bytes")
