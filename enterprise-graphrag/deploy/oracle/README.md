# Oracle Cloud Always Free — the runbook

Free forever, no credit card, always on. 2 OCPU / 12 GB ARM against a measured need of
1,064 MB, so roughly 11x the headroom this service wants.

## What you do vs what I do

| step | who |
|---|---|
| 1. Oracle account + ARM VM | **you** (needs identity verification) |
| 2. open ports in the VCN security list | **you** (web console only) |
| 3. upload the project + 196 MB of data | either |
| 4. `sudo bash bootstrap.sh` | either |
| 5. fill `secrets.env` | **you** — I will not put your Gemini key on a box |
| 6. point the Netlify page at it, verify | **me** |

## 1. The VM

Console → Compute → Instances → Create.

```
Image     Canonical Ubuntu 24.04
Shape     VM.Standard.A1.Flex   (Ampere ARM — this is the free one)
OCPUs     2        Memory  12 GB
Boot vol  50 GB
```

Save the SSH private key it offers. **"Always Free eligible" must appear on the shape** — if
it does not, you have picked an x86 shape and it will bill you.

**Capacity is a lottery.** `Out of host capacity` in popular regions is routine; retry, or
create the instance in a different availability domain. This is the single most common reason
people give up on Oracle free tier.

## 2. Ports — both layers, or nothing works

Oracle blocks inbound traffic in **two** independent places and you must open both:

**VCN security list** (console): Networking → VCN → your subnet → Security List → Add Ingress
Rules. Source `0.0.0.0/0`, TCP, destination ports **80** and **443**.

**The VM's own iptables**: Oracle's Ubuntu images ship an INPUT chain that REJECTs everything
except SSH, persisted across reboots. `bootstrap.sh` handles this one.

Almost every "I opened the port and it still times out" report about Oracle is someone who
did the first and not the second.

## 3–4. Get the code up and run bootstrap

```bash
# from your laptop — 196 MB, a few minutes
scp -i <key> -r GraphRAG-Implementation ubuntu@<IP>:/tmp/src
ssh -i <key> ubuntu@<IP>
sudo mkdir -p /opt/graphrag && sudo mv /tmp/src /opt/graphrag/src
sudo bash /opt/graphrag/src/enterprise-graphrag/deploy/oracle/bootstrap.sh
```

The two large files are gitignored, so a `git clone` on the VM will **not** have them. Copy
them explicitly if you clone rather than scp:

```
enterprise-graphrag/datasets/enterprise/corpus.jsonl           141 MB
enterprise-graphrag/pipeline3_round3/index_v2/edges_prov.json   55 MB
```

The build takes 15–25 minutes on two ARM cores, mostly torch. The Dockerfile picks stock PyPI
torch on aarch64 and the CPU-only index on x86_64, so the same file builds on both.

## 5. Secrets

`bootstrap.sh` writes `/opt/graphrag/secrets.env` with placeholders on its first run, then
stops. Fill it and re-run:

```
GOOGLE_API_KEY=...
TIGERGRAPH_HOST=https://<id>.i.tgcloud.io
TIGERGRAPH_SECRET=...
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=...
DEMO_TOKEN=<long random string>
```

`chmod 600`, never in git.

## 6. HTTPS — not optional

`graphrag-benchmark.netlify.app` is HTTPS, and **a browser will not let an HTTPS page fetch
from an http:// endpoint.** The request is blocked before it leaves the browser, so the query
box would fail for every visitor while the backend logs stay empty and look perfectly healthy.
That is the most confusing possible failure, so TLS is set up from the start.

`Caddyfile` uses **sslip.io** — `graphrag.130-61-24-7.sslip.io` resolves to `130.61.24.7`
with no registration and no DNS, and Let's Encrypt issues for it. A bare VM gets a real
certificate with no domain purchase.

```bash
sudo cp /opt/graphrag/src/enterprise-graphrag/deploy/oracle/Caddyfile /etc/caddy/Caddyfile
sudo sed -i "s/REPLACE-WITH-DASHED-IP/$(curl -s ifconfig.me | tr . -)/" /etc/caddy/Caddyfile
sudo systemctl reload caddy
curl https://graphrag.$(curl -s ifconfig.me | tr . -).sslip.io/api/status
```

Expect `{"state":"ready"}` within ~60 s of the container starting.

## Then send me the URL

I will point `demo/build_static.py` at it, rebuild, redeploy Netlify, and verify a live query
end to end.

## Security notes

- The container binds **127.0.0.1:8090**, not `0.0.0.0`. The only route in is Caddy, which
  terminates TLS and applies CORS. Publishing the port directly would expose the unencrypted
  service and let anyone bypass both.
- CORS allows **only** `https://graphrag-benchmark.netlify.app`. Not `*` — this endpoint
  spends money per call, and a wildcard lets any page on the internet drive it from a
  visitor's browser.
- `DEMO_TOKEN` will be visible in the published page's JavaScript. It stops crawlers, not a
  determined person. The per-IP rate limit is the real spend control.
- `--memory=3g` on the container so a runaway cannot take the whole VM down with it.

## Known gaps

- **The Savanna workspace can suspend.** Live queries then fail while the static numbers stay
  fine. No keep-warm ping is wired up.
- **No CI.** The image is built on the VM by hand.
- **Oracle reclaims idle Always Free VMs.** They define idle as low CPU *and* low network over
  7 days. A service answering the occasional query is usually safe, but a completely unused
  box can be reclaimed.
