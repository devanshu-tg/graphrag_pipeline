# Copy-paste message for IT / platform team

Render only, one option, with the exact line items. Full spec in IT-REQUEST.md if they want it.

---

**Subject: Small container host for a demo backend**

Hi — I have a public GraphRAG demo (three retrieval strategies compared on TigerGraph) whose
results page is already live. I need somewhere to run the backend that answers live queries.

It's one container:

- **2 vCPU, 4 GB RAM, 20 GB disk** — the process sits at ~1 GB resident, 2 GB is the floor
- **No GPU, no database, no persistent storage, no scaling**
- **Outbound HTTPS to two hosts only:** `*.i.tgcloud.io` (our Savanna workspace) and
  `generativelanguage.googleapis.com` (Gemini)
- **Inbound 443**, or internal-only behind VPN/SSO — internal is my preference, not a
  compromise: the endpoint calls a paid LLM API on every request, so a public URL is
  effectively an anonymous endpoint spending our quota
- **Two secrets as environment variables**

**Do we already have somewhere this could live?** A namespace on an existing cluster, a small
VM in one of our cloud accounts, or an internal box would all be ideal. I have a working
Dockerfile and I'll deploy and maintain it — nothing ongoing for your side.

**If there's nothing internal, here is exactly what I'd need on Render — $25/month total:**

| line item | choice | cost |
|---|---|---|
| Workspace plan | **Hobby** | **$0/mo** |
| Web Service compute | **Standard** — 2 GB RAM, 1 CPU | **$25/mo** |
| Deploy method | Docker (I supply the Dockerfile) | — |
| Postgres / Key Value | **none** — we use our existing TigerGraph | $0 |
| Persistent disk | **none**, the service is stateless | $0 |
| Bandwidth | well under the 5 GB included | $0 |

The free and Starter compute tiers are both 512 MB, which can't run this — Standard is the
smallest that fits. Pro (4 GB / 2 CPU) would be more comfortable but isn't necessary.

Two constraints, so nothing unsuitable gets suggested: **requests take up to 80 seconds**
(three pipelines, each making an LLM call — rules out Lambda/API Gateway at 29s and Netlify
Functions at 10s), and it **holds ~1 GB in memory permanently**, so 512 MB tiers can't run it
and scale-to-zero means a 60-second cold start for the first visitor.

Happy to walk through it or hand over the Dockerfile. Thanks!

---

## If they ask what you need to start

- the hostname, or a Render workspace invite
- SSH access, or a registry/pipeline to push the image to
- the two outbound hosts allowed through egress

Live the same day after that.

## One note for you, not for them

Standard is **1 CPU**. Every query reranks a 200-candidate pool through a cross-encoder, which
is CPU-bound, and that was measured on a machine with several cores free. Expect queries slower
than local — comfortably inside Render's 100-minute response limit, just less snappy. If it
feels sluggish, Pro (4 GB / 2 CPU) is the fix.
