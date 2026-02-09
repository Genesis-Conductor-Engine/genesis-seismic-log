import json
import timeit
from datetime import datetime

# System metrics (from Diamond Vault verified logs)
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

# Sample data mimicking /api/bench/live
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

def serialize_indent():
    return json.dumps(data, indent=2)

def serialize_separators():
    return json.dumps(data, separators=(',', ':'))

def benchmark():
    # Warmup
    for _ in range(100):
        serialize_indent()
        serialize_separators()

    iterations = 50000

    start_indent = timeit.default_timer()
    for _ in range(iterations):
        serialize_indent()
    time_indent = timeit.default_timer() - start_indent

    start_sep = timeit.default_timer()
    for _ in range(iterations):
        serialize_separators()
    time_sep = timeit.default_timer() - start_sep

    size_indent = len(serialize_indent().encode('utf-8'))
    size_sep = len(serialize_separators().encode('utf-8'))

    print(f"Iterations: {iterations}")
    print(f"Indent=2: {time_indent:.4f}s, Size: {size_indent} bytes")
    print(f"Separators: {time_sep:.4f}s, Size: {size_sep} bytes")

    if time_sep > 0:
        speedup = time_indent / time_sep
    else:
        speedup = 0

    size_reduction = (size_indent - size_sep) / size_indent * 100

    print(f"Speedup: {speedup:.2f}x")
    print(f"Size Reduction: {size_reduction:.2f}%")

if __name__ == "__main__":
    benchmark()
