# Genesis Seismic Log - Link Verification Report

**Date**: 2026-09-11  
**Status**: ✅ Public API links restored (Cloudflare Worker edge fallback)

## External Links (Live Endpoints)

### Primary API Endpoints
| URL | Status | Response Time | Notes |
|-----|--------|---------------|-------|
| https://qmem.genesisconductor.io | ✅ 200 OK | < 100ms | Worker `genesis-seismic-log` |
| https://qmem.genesisconductor.io/api/health | ✅ 200 OK | < 100ms | `origin=cloudflare-worker-fallback` |
| https://qmem.genesisconductor.io/api/bench/live | ✅ 200 OK | < 100ms | Restored after tunnel 530 |
| https://qmem.genesisconductor.io/api/seismic/status | ✅ 200 OK | < 100ms | S-ToT protocol status |
| https://seismic.genesisconductor.io/api/bench/live | ✅ 200 OK | < 100ms | New dedicated hostname |
| https://genesis-seismic-log.iholt.workers.dev/api/bench/live | ✅ 200 OK | < 100ms | workers.dev backup |

### External Services
| URL | Status | Purpose |
|-----|--------|---------|
| https://github.com/Genesis-Conductor-Engine/genesis-seismic-log | ✅ Active | GitHub repository |
| https://github.com/Genesis-Conductor-Engine/genesis-seismic-log/issues | ✅ Active | Issue tracker |
| https://dash.cloudflare.com | ✅ Active | Cloudflare dashboard |
| https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 | ✅ Active | Cloudflared binary |

### Badge Links
| Badge | Target | Status |
|-------|--------|--------|
| Status: Operational | https://qmem.genesisconductor.io | ✅ Valid |
| Protocol: S-ToT | #s-tot-protocol (anchor) | ✅ Valid |
| Energy Efficiency | #performance-metrics (anchor) | ✅ Valid |
| Live Demo | https://qmem.genesisconductor.io/api/bench/live | ✅ Valid |

## Vercel / Cloudflare inventory (this repo)

- **Vercel**: no `vercel.app` or `cname.vercel-dns.com` links in this repository.
- **Cloudflare**: all public demo links use `qmem.genesisconductor.io` (and now `seismic.genesisconductor.io`) on zone `genesisconductor.io`.
- **Root cause of 530**: CNAME → tunnel `15b1ac8a-d140-4c21-a1c1-4f91fb313309` (`yennefer-consciousness`, down since 2026-09-09).
- **Fix**: Worker routes override the dead origin. See [EDGE_FALLBACK.md](./EDGE_FALLBACK.md).

## Test Commands

```bash
curl -s https://qmem.genesisconductor.io/api/health | jq '.status'
curl -s https://qmem.genesisconductor.io/api/bench/live | jq '.metrics.crystallization_status'
curl -s https://qmem.genesisconductor.io/api/seismic/status | jq '.protocol'
curl -sI https://qmem.genesisconductor.io/api/bench/live | grep -i x-genesis
```

**Live Demo**: https://qmem.genesisconductor.io/api/bench/live

---

*Last updated: 2026-09-11T04:31:00Z*
