# Copy-paste message for IT / platform team

Short on purpose. The full spec is in IT-REQUEST.md if they ask for detail.

---

**Subject: Small container host for a public GraphRAG demo**

Hi — I've built a public demo comparing three retrieval strategies (LLM-only, vector RAG, and
GraphRAG on TigerGraph) over a 20k-document corpus. The results page is already live on
Netlify. What I need now is somewhere to run the backend that answers live queries.

It's one small container:

- **2 vCPU, 4 GB RAM, 20 GB disk** (the process sits at ~1 GB resident; 2 GB is the floor)
- **No GPU, no database, no persistent storage, no scaling**
- **Outbound HTTPS to two hosts only:** our TigerGraph Savanna workspace (`*.i.tgcloud.io`)
  and the Gemini API (`generativelanguage.googleapis.com`)
- **Inbound 443** — or internal-only behind VPN/SSO, which I'd actually prefer
- **Two secrets as environment variables** (a Gemini API key and our existing Savanna secret)

**Do we already have somewhere this could live?** A namespace on an existing cluster, a small
VM in one of our cloud accounts, or an internal box would all be ideal — I have a working
Dockerfile and I'll deploy and maintain it, so there's nothing ongoing for your side.

**Internal-only is my preference, not a compromise.** The endpoint calls a paid LLM API on
every request, so a public URL is effectively an anonymous endpoint spending our quota. It has
a shared token and a rate limit in front of it, but behind SSO that concern disappears
entirely.

**If there's genuinely nothing internal:** Render would work. The workspace plan is free
(Hobby) and we'd pay only for one *Standard* compute instance at 2 GB — I'll confirm the exact
figure. Fly.io is roughly $14/month for the same specs if cost matters more than familiarity.
Either is single-digit-to-low-double-digit dollars a month.

Two things I should flag so nobody suggests a cheaper shape that won't work:

- **A request takes up to 80 seconds** (three retrieval pipelines, each making an LLM call), so
  anything with a short gateway timeout is out — Lambda/API Gateway at 29s, Netlify Functions
  at 10s.
- **It holds ~1 GB in memory permanently** (a parsed corpus and a graph edge index), so 512 MB
  free tiers can't run it and scale-to-zero means a 60-second cold start for the first visitor.

Happy to walk through it or hand over the Dockerfile. Thanks!

---

## If they ask "what do you need from us to start"

- a hostname or IP
- SSH access, or a registry/pipeline to push the image to
- the two outbound hosts allowed through egress rules

It's live the same day after that.
