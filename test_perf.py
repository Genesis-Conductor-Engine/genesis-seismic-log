import json
import timeit

data = {
    "timestamp": "2023-10-27T10:00:00.000Z",
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

def indented():
    return json.dumps(data, indent=2).encode()

def tight():
    return json.dumps(data, separators=(',', ':')).encode()

print("Indented: ", timeit.timeit(indented, number=100000))
print("Tight:    ", timeit.timeit(tight, number=100000))
