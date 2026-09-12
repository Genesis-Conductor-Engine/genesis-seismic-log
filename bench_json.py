
import json
import timeit
import sys
from simple_seismic_server import SYSTEM_METRICS
from datetime import datetime
import time

# Reconstruct data objects
def get_root_data():
    return {
        "service": "Genesis Seismic Log",
        "version": "1.0.0",
        "status": "operational",
        "protocol": "S-ToT (Seismic Tree-of-Thoughts)",
        "endpoints": {
            "live": "/api/bench/live",
            "health": "/api/health",
            "seismic": "/api/seismic/status"
        }
    }

def get_bench_data():
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

def get_seismic_data():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "protocol": "Seismic Tree-of-Thoughts (S-ToT)",
        "phases": {
            "quantum_branching": {
                "status": "complete",
                "branches_generated": 3,
                "orthogonality_score": 0.94
            },
            "seismography": {
                "status": "complete",
                "stress_factor": 0.1,
                "perturbations_applied": 1000,
                "shake_intensity": "thermal_langevin"
            },
            "crystallization": {
                "status": "CRYSTALLINE",
                "threshold": 1e-4,
                "measured_divergence": 3.2e-5,
                "invariance_score": 0.998
            },
            "cold_snap": {
                "status": "complete",
                "branches_shattered": 0,
                "branches_crystalline": 3,
                "synthesis": "unanimous_convergence"
            }
        },
        "landauer_limit": {
            "measured_joules_per_op": 0.042,
            "theoretical_minimum": 0.0029,
            "efficiency_percentage": 6.9
        }
    }

datasets = {
    "root": get_root_data(),
    "bench": get_bench_data(),
    "seismic": get_seismic_data()
}

def benchmark():
    print(f"{'Endpoint':<10} | {'Method':<15} | {'Size (bytes)':<12} | {'Time (us/op)':<12}")
    print("-" * 60)

    for name, data in datasets.items():
        # Baseline (Indent=2)
        serialized_indent = json.dumps(data, indent=2).encode('utf-8')
        size_indent = len(serialized_indent)

        t_indent = timeit.timeit(lambda: json.dumps(data, indent=2), number=10000)
        avg_indent = (t_indent / 10000) * 1e6

        print(f"{name:<10} | {'indent=2':<15} | {size_indent:<12} | {avg_indent:.2f}")

        # Optimized (Separators)
        serialized_opt = json.dumps(data, separators=(',', ':')).encode('utf-8')
        size_opt = len(serialized_opt)

        t_opt = timeit.timeit(lambda: json.dumps(data, separators=(',', ':')), number=10000)
        avg_opt = (t_opt / 10000) * 1e6

        print(f"{name:<10} | {'separators':<15} | {size_opt:<12} | {avg_opt:.2f}")

        # Improvement
        size_diff = size_indent - size_opt
        size_pct = (size_diff / size_indent) * 100
        time_diff = avg_indent - avg_opt
        time_pct = (time_diff / avg_indent) * 100

        print(f"{'':<10} | {'IMPROVEMENT':<15} | {size_diff} ({size_pct:.1f}%) | {time_diff:.2f} ({time_pct:.1f}%)")
        print("-" * 60)

if __name__ == "__main__":
    benchmark()
