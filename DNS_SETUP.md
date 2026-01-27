# DNS Configuration for seismic.genesisconductor.io

## Current Status

✅ Cloudflare tunnel configured (tunnel ID: `15b1ac8a-d140-4c21-a1c1-4f91fb313309`)
✅ Seismic API server running on `localhost:8003`
✅ Ingress rule added to `/etc/cloudflared/config.yml`
⏳ **DNS CNAME record needed**

## Required DNS Configuration

To make `seismic.genesisconductor.io` publicly accessible, add a CNAME record in Cloudflare DNS.

### Step-by-Step Instructions

#### 1. Log in to Cloudflare Dashboard

Visit: [https://dash.cloudflare.com](https://dash.cloudflare.com)

#### 2. Select Your Domain

Click on `genesisconductor.io` in your domain list.

#### 3. Navigate to DNS Settings

Click the **DNS** tab in the left sidebar.

#### 4. Add CNAME Record

Click **+ Add record** and enter:

| Field | Value |
|-------|-------|
| **Type** | CNAME |
| **Name** | `seismic` |
| **Target** | `15b1ac8a-d140-4c21-a1c1-4f91fb313309.cfargotunnel.com` |
| **Proxy status** | ✅ Proxied (orange cloud) |
| **TTL** | Auto |

#### 5. Save Record

Click **Save** to create the CNAME record.

### Verification

After adding the DNS record (propagation takes 1-5 minutes):

```bash
# Test DNS resolution
nslookup seismic.genesisconductor.io

# Test health endpoint
curl https://seismic.genesisconductor.io/api/health

# Test benchmark metrics
curl https://seismic.genesisconductor.io/api/bench/live | jq
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-27T...",
  "services": {
    "seismic_wrapper": "active",
    "qmem_bridge": "active",
    "crystallization_verifier": "active"
  }
}
```

## Alternative: Use Existing qmem Subdomain

If you want to test immediately without waiting for DNS:

The `qmem.genesisconductor.io` subdomain already points to port 8003:

```bash
# Test with existing subdomain
curl https://qmem.genesisconductor.io/api/health
curl https://qmem.genesisconductor.io/api/bench/live | jq
```

Both `seismic.genesisconductor.io` and `qmem.genesisconductor.io` will serve the same Seismic Log API.

## Troubleshooting

### DNS Not Resolving

- **Wait 5 minutes**: DNS propagation can take time
- **Clear DNS cache**: `sudo systemd-resolve --flush-caches` (Linux)
- **Check Cloudflare DNS**: Verify CNAME record exists in dashboard
- **Test with dig**: `dig seismic.genesisconductor.io`

### Connection Refused

- **Verify API is running**: `curl http://localhost:8003/api/health`
- **Check cloudflared status**: `systemctl status cloudflared`
- **Review tunnel logs**: `sudo journalctl -u cloudflared -f`

### 502 Bad Gateway

- **API server stopped**: Restart with `python3 simple_seismic_server.py &`
- **Port mismatch**: Verify API is on port 8003 (check `lsof -i :8003`)
- **Firewall blocking**: Ensure localhost connections allowed

## Cloudflare Tunnel Configuration

Current configuration (`/etc/cloudflared/config.yml`):

```yaml
tunnel: 15b1ac8a-d140-4c21-a1c1-4f91fb313309
credentials-file: /etc/cloudflared/credentials.json

ingress:
  - hostname: yennefer.genesisconductor.io
    service: http://localhost:8000
  - hostname: dashboard.genesisconductor.io
    service: http://localhost:8080
  - hostname: soul.genesisconductor.io
    service: http://localhost:8088
  - hostname: qmem.genesisconductor.io
    service: http://localhost:8003
  - hostname: seismic.genesisconductor.io
    service: http://localhost:8003
  - hostname: mcp.genesisconductor.io
    service: http://localhost:8096
  - service: http_status:404
```

## Summary

Once the CNAME record is added:

✅ `https://seismic.genesisconductor.io/api/health` → Health check
✅ `https://seismic.genesisconductor.io/api/bench/live` → Live metrics
✅ `https://seismic.genesisconductor.io/api/seismic/status` → S-ToT protocol status

Ready to share with Extropic/Tesla! 🚀
