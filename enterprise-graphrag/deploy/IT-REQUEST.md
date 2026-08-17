# Hosting request — GraphRAG demo backend

Forward the section below. It is deliberately small: no GPU, no database, no persistent
storage, no scaling, no on-call. One container that answers HTTP requests.

---

## The ask

I need somewhere to run one small Linux container for a public GraphRAG demo. It is the
backend for a benchmark page comparing three retrieval strategies on TigerGraph.

**Preferred, in order — whichever you already have:**

1. **A namespace on an existing container platform** (Kubernetes, ECS, Cloud Run, Nomad).
   I have a working Dockerfile; the image is built and tested.
2. **A small VM with Docker** in a company cloud account.
3. **An internal VM**, reachable over VPN/SSO only — see the note on secrets below, this is
   actually the option I'd prefer.

**Resources** — measured from a running instance, not estimated:

| | |
|---|---|
| CPU | 2 vCPU |
| RAM | **4 GB** (process is 1,064 MB resident; 2 GB is the floor, 4 GB gives headroom) |
| Disk | 20 GB (3 GB image + 200 MB data) |
| GPU | none |
| Persistent volume | none — the service is stateless |
| Database | none — it talks to our existing TigerGraph Savanna workspace |

**Network:**

- **Outbound HTTPS** to exactly two hosts:
  - `*.i.tgcloud.io` — our TigerGraph Savanna workspace
  - `generativelanguage.googleapis.com` — Gemini API
- **Inbound HTTPS on 443.** Either a public DNS name with a certificate, or internal-only
  behind SSO/VPN. Internal-only is fine and preferable.

**Two constraints that rule out the obvious cheap options:**

- **A request takes up to 80 seconds.** Three retrieval pipelines run per question, each
  making an LLM call. This rules out anything with a short gateway timeout — API Gateway at
  29s, Netlify Functions at 10s, Cloudflare Workers, Vercel hobby at 60s.
- **It holds ~1 GB in memory permanently** (a parsed document corpus and a graph edge index).
  512 MB free tiers cannot run it, and scale-to-zero means a 60-second cold start on the
  first request after idle.

**Secrets** — two, needed as environment variables, never in the image:

- `GOOGLE_API_KEY` (Gemini)
- `TIGERGRAPH_SECRET` (our existing workspace)

**Why internal-only is the better option:** the endpoint calls a paid LLM API on every
request. Public, that is an anonymous endpoint spending our quota. It has a shared token and
a 20-request/hour/IP limit in front of it, but behind SSO the problem disappears entirely
rather than being mitigated.

**Cost:** this is roughly a `t3.small` / `e2-small` / 2 GB droplet — single-digit dollars a
month, or free if it lands on existing capacity.

**Effort for you:** none after provisioning. I deploy and maintain it. `docker run` and a
reverse-proxy rule.

---

## What to tell them if asked "why not serverless"

Because the request takes 80 seconds and the process holds a gigabyte of index in memory.
Serverless platforms are built for the opposite shape — short, small, stateless. This is a
long-running query engine with a warm cache. It is the wrong workload for Lambda and the
right workload for one small always-on container.

## What I need back

- the hostname or IP
- SSH access, or a pipeline/registry to push the image to
- confirmation of the two outbound hosts being allowed

Then it is live the same day. Everything is already built and verified:
`enterprise-graphrag/deploy/Dockerfile`.
