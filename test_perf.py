import json
import timeit

data = {
    "status": "healthy",
    "timestamp": "2023-10-25T12:00:00.000000",
    "uptime_seconds": 12345,
    "services": {
        "seismic_wrapper": "active",
        "qmem_bridge": "active",
        "crystallization_verifier": "active"
    }
}

def indented():
    return json.dumps(data, indent=2).encode()

def compact():
    return json.dumps(data, separators=(',', ':')).encode()

if __name__ == "__main__":
    n = 100000
    t_indented = timeit.timeit(indented, number=n)
    t_compact = timeit.timeit(compact, number=n)
    print(f"Indented: {t_indented:.4f} seconds")
    print(f"Compact: {t_compact:.4f} seconds")
    print(f"Speedup: {t_indented/t_compact:.2f}x")

    s_indented = len(indented())
    s_compact = len(compact())
    print(f"Size indented: {s_indented} bytes")
    print(f"Size compact: {s_compact} bytes")
    print(f"Size reduction: {100 * (s_indented - s_compact) / s_indented:.1f}%")
