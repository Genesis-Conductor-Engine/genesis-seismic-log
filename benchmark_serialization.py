import json
import time
from datetime import datetime

# Mock System Metrics
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

# Construct a representative large JSON payload
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

ITERATIONS = 10000

# Benchmark Baseline (Indent=2)
start_time = time.time()
for _ in range(ITERATIONS):
    res = json.dumps(data, indent=2).encode()
baseline_duration = time.time() - start_time
baseline_size = len(res)

# Benchmark Optimized (Separators=(',', ':'))
start_time = time.time()
for _ in range(ITERATIONS):
    res = json.dumps(data, separators=(',', ':')).encode()
optimized_duration = time.time() - start_time
optimized_size = len(res)

print(f"Benchmark Results (over {ITERATIONS} iterations):")
print(f"Baseline (indent=2): {baseline_duration:.4f}s, Size: {baseline_size} bytes")
print(f"Optimized (separators=(',', ':')): {optimized_duration:.4f}s, Size: {optimized_size} bytes")

improvement_time = (baseline_duration - optimized_duration) / baseline_duration * 100
improvement_size = (baseline_size - optimized_size) / baseline_size * 100

print(f"Improvement Time: {improvement_time:.2f}% faster")
print(f"Improvement Size: {improvement_size:.2f}% smaller")
