import json
import timeit
import sys
from datetime import datetime

# System metrics (from simple_seismic_server.py)
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

# Construct the payload
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

def benchmark_indent():
    json.dumps(data, indent=2)

def benchmark_separators():
    json.dumps(data, separators=(',', ':'))

if __name__ == "__main__":
    iterations = 100000

    print(f"Benchmarking JSON serialization with {iterations} iterations...")

    # Baseline (indent=2)
    time_indent = timeit.timeit(benchmark_indent, number=iterations)
    size_indent = len(json.dumps(data, indent=2).encode('utf-8'))

    # Optimized (separators=(',', ':'))
    time_separators = timeit.timeit(benchmark_separators, number=iterations)
    size_separators = len(json.dumps(data, separators=(',', ':')).encode('utf-8'))

    print(f"\nBaseline (indent=2):")
    print(f"  Time: {time_indent:.4f} seconds")
    print(f"  Size: {size_indent} bytes")

    print(f"\nOptimized (separators=(',', ':')):")
    print(f"  Time: {time_separators:.4f} seconds")
    print(f"  Size: {size_separators} bytes")

    speedup = time_indent / time_separators
    size_reduction = (size_indent - size_separators) / size_indent * 100

    print(f"\nResults:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Size Reduction: {size_reduction:.2f}%")
