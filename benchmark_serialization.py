import timeit
import json
from datetime import datetime
import sys

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

def create_bench_response():
    # Mimic the response construction
    return {
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

def benchmark():
    data = create_bench_response()
    iterations = 10000

    def test_indent():
        json.dumps(data, indent=2).encode()

    def test_compact():
        json.dumps(data, separators=(',', ':')).encode()

    # Measure with indentation (unoptimized)
    time_bad = timeit.timeit(test_indent, number=iterations)
    avg_time_bad_ms = (time_bad / iterations) * 1000
    size_bad = len(json.dumps(data, indent=2).encode())

    # Measure with separators (optimized)
    time_good = timeit.timeit(test_compact, number=iterations)
    avg_time_good_ms = (time_good / iterations) * 1000
    size_good = len(json.dumps(data, separators=(',', ':')).encode())

    print(f"Scenario A (indent=2): Avg Time={avg_time_bad_ms:.4f}ms, Size={size_bad} bytes")
    print(f"Scenario B (separators): Avg Time={avg_time_good_ms:.4f}ms, Size={size_good} bytes")

    improvement_time = (avg_time_bad_ms - avg_time_good_ms) / avg_time_bad_ms * 100
    improvement_size = (size_bad - size_good) / size_bad * 100

    print(f"Performance Improvement: {improvement_time:.2f}% faster")
    print(f"Size Reduction: {improvement_size:.2f}% smaller")

if __name__ == "__main__":
    benchmark()
