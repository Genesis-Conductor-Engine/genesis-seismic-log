#!/usr/bin/env python3
import json
import timeit
import sys
from datetime import datetime

# Import SYSTEM_METRICS from the server file
try:
    from simple_seismic_server import SYSTEM_METRICS
except ImportError:
    print("Error: Could not import simple_seismic_server.py")
    sys.exit(1)

def get_data():
    """Generates the data structure used in send_json for /api/bench/live endpoint"""
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
    data = get_data()
    iterations = 10000

    # Baseline: indent=2
    t_baseline = timeit.timeit(lambda: json.dumps(data, indent=2), number=iterations)
    size_baseline = len(json.dumps(data, indent=2).encode('utf-8'))

    # Optimized: separators=(',', ':')
    t_optimized = timeit.timeit(lambda: json.dumps(data, separators=(',', ':')), number=iterations)
    size_optimized = len(json.dumps(data, separators=(',', ':')).encode('utf-8'))

    # Further Optimized: separators=(',', ':'), check_circular=False
    t_super = timeit.timeit(lambda: json.dumps(data, separators=(',', ':'), check_circular=False), number=iterations)
    size_super = len(json.dumps(data, separators=(',', ':'), check_circular=False).encode('utf-8'))

    print(f"Benchmark Results ({iterations} iterations):")
    print("-" * 60)
    print(f"Baseline (indent=2):")
    print(f"  Time: {t_baseline:.4f} s")
    print(f"  Size: {size_baseline} bytes")
    print("-" * 60)
    print(f"Optimized (separators=(',', ':')):")
    print(f"  Time: {t_optimized:.4f} s")
    print(f"  Size: {size_optimized} bytes")
    print("-" * 60)
    print(f"Super Optimized (separators + check_circular=False):")
    print(f"  Time: {t_super:.4f} s")
    print(f"  Size: {size_super} bytes")
    print("-" * 60)

    time_improvement = (t_baseline - t_optimized) / t_baseline * 100
    super_improvement = (t_baseline - t_super) / t_baseline * 100

    print(f"Improvement (Optimized):")
    print(f"  Time: {time_improvement:.2f}% faster")
    print(f"Improvement (Super):")
    print(f"  Time: {super_improvement:.2f}% faster")

if __name__ == "__main__":
    benchmark()
