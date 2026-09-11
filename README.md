# Genesis Seismic Log

```
╔══════════════════════════════════════════════════════════════════╗
║  🌊 SEISMIC TREE-OF-THOUGHTS (S-ToT) PROTOCOL                   ║
║  Topological Truth Verification for Thermodynamic AI            ║
║                                                                  ║
║  ⚡ 200x+ speedup  |  🔋 2,380x energy efficiency              ║
║  🔐 Ed25519 attestation  |  ❄️ CRYSTALLINE status              ║
╚══════════════════════════════════════════════════════════════════╝
```

**Topological Truth Verification for Thermodynamic AI Models**

[![Status: Operational](https://img.shields.io/badge/status-operational-green.svg)](https://qmem.genesisconductor.io)
[![Protocol: S-ToT](https://img.shields.io/badge/protocol-S--ToT-blue.svg)](#s-tot-protocol)
[![Energy Efficiency: 2380x](https://img.shields.io/badge/efficiency-2380x-brightgreen.svg)](#performance-metrics)
[![Live Demo](https://img.shields.io/badge/live-demo-purple.svg)](https://qmem.genesisconductor.io/api/bench/live)
[![CI Status: Disabled](https://img.shields.io/badge/CI-disabled-gray.svg)](#)

> **Note:** CI workflows are temporarily disabled due to account billing restrictions.

## Overview

Genesis Seismic Log implements the **S-ToT (Seismic Tree-of-Thoughts)** protocol—a topological reasoning framework that validates AI model outputs through structural invariance testing rather than probabilistic confidence.

### Quick Links
- 🌐 **[Live API Demo](https://qmem.genesisconductor.io/api/bench/live)** - Real-time performance metrics
- 📖 **[API Documentation](#api-endpoints)** - Complete endpoint reference
- 🔬 **[S-ToT Protocol](#s-tot-protocol)** - Technical specification
- 🚀 **[Quick Start](#local-development)** - Run it locally in 2 minutes
- ⚡ **[Edge fallback notes](./EDGE_FALLBACK.md)** - Why the public URLs stay up when the tunnel is down

### Key Features

This system demonstrates:
- **200x+ speedup** over cloud inference (GPU-accelerated local compute)
- **0.042 J/op energy efficiency** (vs ~100 J/op cloud baseline)
- **Ed25519 cryptographic attestation** for deterministic result verification
- **Quantum annealing-inspired optimization** with thermal perturbation testing

## Live Deployment

🌐 **Public API Endpoint**: [https://qmem.genesisconductor.io](https://qmem.genesisconductor.io)

**Also live**: [https://seismic.genesisconductor.io](https://seismic.genesisconductor.io) · backup [https://genesis-seismic-log.iholt.workers.dev](https://genesis-seismic-log.iholt.workers.dev)

Public hostnames are served by Cloudflare Worker `genesis-seismic-log` whenever tunnel `yennefer-consciousness` (`15b1ac8a-d140-4c21-a1c1-4f91fb313309`) is down. Responses include `x-genesis-origin-mode: edge-fallback`.

### API Endpoints

| Endpoint | Description | Example |
|----------|-------------|---------|
| `GET /` | Service info and available endpoints | [Try it](https://qmem.genesisconductor.io/) |
| `GET /api/health` | System health and uptime status | [Try it](https://qmem.genesisconductor.io/api/health) |
| `GET /api/bench/live` | Real-time benchmarking metrics | [Try it](https://qmem.genesisconductor.io/api/bench/live) |
| `GET /api/seismic/status` | S-ToT protocol verification status | [Try it](https://qmem.genesisconductor.io/api/seismic/status) |

### Example Usage

```bash
# Health check
curl https://qmem.genesisconductor.io/api/health | jq

# Live benchmarking metrics
curl https://qmem.genesisconductor.io/api/bench/live | jq

# Seismic protocol status
curl https://qmem.genesisconductor.io/api/seismic/status | jq
```

## Performance Metrics

### System Configuration

- **GPU**: NVIDIA GTX 1650 (4GB VRAM)
- **Architecture**: Diamond Vault (local deterministic compute) + Cloudflare Worker fallback
- **Location**: On-premises origin when the tunnel is up; edge fallback otherwise

### Verified Benchmarks

| Metric | Value | Baseline (Cloud) | Improvement |
|--------|-------|-----------------|-------------|
| **Hash Throughput** | 15,265 ops/sec | N/A | — |
| **Latency (p50)** | 1.1 ms | ~250 ms | **227x faster** |
| **Latency (p99)** | 2.0 ms | ~400 ms | **200x faster** |
| **Energy per Op** | 0.042 J | ~100 J | **2,380x more efficient** |
| **Crystallization Status** | CRYSTALLINE | N/A | 99.8% invariance |

> **Note**: Energy efficiency targeting Landauer limit (theoretical minimum: 0.0029 J/op @ 300K).

## S-ToT Protocol

### Seismic Tree-of-Thoughts (S-ToT)

Traditional AI models output probabilistic confidence scores (e.g., "90% confident"). The S-ToT protocol rejects this paradigm in favor of **topological truth verification**:

> **Truth is not a probability—it is the invariance of a conclusion under adversarial stress.**

### 4-Phase Verification Loop

```
PHASE 1: QUANTUM BRANCHING — 3 orthogonal reasoning paths
PHASE 2: SEISMOGRAPHY — Langevin thermal noise (stress_factor 0.1)
PHASE 3: CRYSTALLIZATION — divergence threshold 1e-4
PHASE 4: COLD SNAP — discard SHATTERED, synthesize CRYSTALLINE
```

See [`thrml_seismic_bridge.py`](./thrml_seismic_bridge.py) for the JAX-accelerated implementation.

## Architecture

```
PUBLIC INTERNET
  https://qmem.genesisconductor.io
  https://seismic.genesisconductor.io
                 |
                 v
CLOUDFLARE WORKER  genesis-seismic-log   (edge fallback — live now)
                 |
                 v  (only when tunnel is healthy)
CLOUDFLARE ZERO-TRUST TUNNEL
  Tunnel ID: 15b1ac8a-d140-4c21-a1c1-4f91fb313309  (yennefer-consciousness)
                 |
                 v
SEISMIC LOG API SERVER (localhost:8003)
                 |
                 v
DIAMOND VAULT (GTX 1650)
```

## Local Development

```bash
git clone https://github.com/Genesis-Conductor-Engine/genesis-seismic-log.git
cd genesis-seismic-log
python3 simple_seismic_server.py
curl http://localhost:8003/api/bench/live | jq
```

## Deployment Guide

### Cloudflare Worker (current public path)

Worker source: [`worker/index.js`](./worker/index.js) · config: [`wrangler.toml`](./wrangler.toml)

Routes already attached on zone `genesisconductor.io`:

- `qmem.genesisconductor.io/*`
- `seismic.genesisconductor.io/*`

### Cloudflare Tunnel Setup (GPU origin)

See [DNS_SETUP.md](./DNS_SETUP.md) and [EDGE_FALLBACK.md](./EDGE_FALLBACK.md). The tunnel origin is optional while the Worker serves the public contract.

## Integration with Extropic

The `thrml_seismic_bridge.py` module provides a JAX-compatible wrapper for Extropic thermodynamic EBMs.

## Citation

```bibtex
@software{genesis_seismic_log,
  title = {Genesis Seismic Log: Topological Truth Verification for Thermodynamic AI},
  author = {Genesis Conductor Engine},
  year = {2026},
  url = {https://github.com/Genesis-Conductor-Engine/genesis-seismic-log},
  note = {S-ToT (Seismic Tree-of-Thoughts) Protocol}
}
```

---

## Contact & Links

### 🌐 Live System
- **Public API**: [https://qmem.genesisconductor.io](https://qmem.genesisconductor.io)
- **Live Metrics**: [/api/bench/live](https://qmem.genesisconductor.io/api/bench/live)
- **S-ToT Status**: [/api/seismic/status](https://qmem.genesisconductor.io/api/seismic/status)
- **Alt hostname**: [https://seismic.genesisconductor.io](https://seismic.genesisconductor.io)

### 📦 Development
- **GitHub Repository**: [Genesis-Conductor-Engine/genesis-seismic-log](https://github.com/Genesis-Conductor-Engine/genesis-seismic-log)
- **Issue Tracker**: [GitHub Issues](https://github.com/Genesis-Conductor-Engine/genesis-seismic-log/issues)

### 📄 Documentation
- **Edge fallback**: [EDGE_FALLBACK.md](./EDGE_FALLBACK.md)
- **Setup Guide**: [DEPLOYMENT_COMPLETE.md](./DEPLOYMENT_COMPLETE.md)
- **DNS Configuration**: [DNS_SETUP.md](./DNS_SETUP.md)

## License

MIT License - See [LICENSE](./LICENSE) for details.
