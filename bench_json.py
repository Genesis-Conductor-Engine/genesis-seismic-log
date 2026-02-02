import timeit
import json
import sys
# Add current directory to path so we can import simple_seismic_server
sys.path.append('.')
from simple_seismic_server import SYSTEM_METRICS

# Reconstruct the complex data structure used in /api/bench/live
data = {
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

def bench_indent():
    json.dumps(data, indent=2)

def bench_separators():
    json.dumps(data, separators=(',', ':'))

if __name__ == "__main__":
    iterations = 100000

    print(f"Benchmarking JSON serialization with {iterations} iterations...")

    # Measure Size
    size_indent = len(json.dumps(data, indent=2).encode('utf-8'))
    size_separators = len(json.dumps(data, separators=(',', ':')).encode('utf-8'))

    print(f"Size (indent=2): {size_indent} bytes")
    print(f"Size (separators): {size_separators} bytes")
    print(f"Reduction: {size_indent - size_separators} bytes ({((size_indent - size_separators) / size_indent) * 100:.2f}%)")

    # Measure Time
    time_indent = timeit.timeit(bench_indent, number=iterations)
    time_separators = timeit.timeit(bench_separators, number=iterations)

    print(f"\nTime (indent=2): {time_indent:.4f}s")
    print(f"Time (separators): {time_separators:.4f}s")

    if time_separators > 0:
        print(f"Speedup: {time_indent / time_separators:.2f}x")
