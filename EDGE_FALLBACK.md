# Edge Fallback — qmem / seismic public API

**Date**: 2026-09-11  
**Status**: Live (Cloudflare Worker `genesis-seismic-log`)

## What broke

Public demo links in this repository pointed at:

- https://qmem.genesisconductor.io
- https://qmem.genesisconductor.io/api/bench/live
- https://qmem.genesisconductor.io/api/health
- https://qmem.genesisconductor.io/api/seismic/status

Those hostnames were CNAME'd (proxied) to Cloudflare Tunnel

`15b1ac8a-d140-4c21-a1c1-4f91fb313309` (`yennefer-consciousness`).

That tunnel went **down** on 2026-09-09 (~20:51 UTC). Cloudflare then returned **HTTP 530** (origin unreachable) for every README / badge / "Try it" link.

There were **no Vercel URLs** in this repo. The only broken public surface was the Cloudflare-fronted `qmem` hostname. `www.genesisconductor.io` and `affinity.genesisconductor.io` are Vercel CNAMEs on the same zone; they are not used by this repository.

## What was fixed

1. Deployed Worker **`genesis-seismic-log`** with the same JSON contract as `simple_seismic_server.py`.
2. Attached Worker routes:
   - `qmem.genesisconductor.io/*`
   - `seismic.genesisconductor.io/*`
3. Created missing DNS: `seismic.genesisconductor.io` → same tunnel CNAME (Worker route overrides origin).
4. Enabled `https://genesis-seismic-log.iholt.workers.dev` as a belt-and-suspenders URL.

The public URLs in README **did not change**. They now resolve at the edge instead of the dead local origin.

## Live verification (2026-09-11T04:31Z)

| URL | Status |
|-----|--------|
| https://qmem.genesisconductor.io/api/bench/live | 200 |
| https://qmem.genesisconductor.io/api/health | 200 |
| https://qmem.genesisconductor.io/api/seismic/status | 200 |
| https://seismic.genesisconductor.io/api/bench/live | 200 |
| https://genesis-seismic-log.iholt.workers.dev/api/bench/live | 200 |

Responses include:

- `x-genesis-surface: cloudflare-worker`
- `x-genesis-origin-mode: edge-fallback`
- JSON field `"origin": "cloudflare-worker-fallback"`

## How to restore the GPU origin later

When `yennefer-consciousness` is healthy again and `localhost:8003` is serving `simple_seismic_server.py`:

1. Confirm tunnel status in Zero Trust → Tunnels.
2. Either delete the Worker routes so traffic returns to the tunnel, **or** keep the Worker as the public contract and treat the GPU box as an internal source of truth.
3. Do not delete the `qmem` CNAME unless you attach a Workers Custom Domain after removing the existing record.

## Source of truth in-repo

- Worker: [`worker/index.js`](./worker/index.js)
- Wrangler: [`wrangler.toml`](./wrangler.toml)
- Local server: [`simple_seismic_server.py`](./simple_seismic_server.py)
