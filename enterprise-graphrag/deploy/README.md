# Deploying the live half

The static site (`docs/`, on Netlify) needs nothing. This is the other half — the query
console that actually runs the three pipelines — and it needs a container, 2 GB of RAM, and
two API secrets.

## Why it cannot go on Netlify

| | |
|---|---|
| a three-arm query takes | **~77 seconds** |
| Netlify Functions cap at | 10 s (26 s background) |
| resident memory once warm | **1,064 MB** |
| Netlify Functions cap at | 1 GB, 50 MB bundle |
| runtime data | 196 MB (corpus + edge index) |

Every number above is measured on this machine, not estimated. Netlify hosts the static
site well and cannot host this at all. Fly.io is the default here because it imposes no
HTTP timeout short enough to matter; Railway and Render also work.

## What it costs

One always-on 2 GB shared-CPU machine, roughly **$5–10/month**, plus Gemini tokens per
query. `min_machines_running = 1` is deliberate: cold start is ~60 s (141 MB corpus parsed
into a dict, 55 MB edge index, two models), and scale-to-zero means the first visitor after
a quiet period stares at a blank page for a minute.

## Before you start

The build needs two files that are `.gitignored` because GitHub rejects anything over
100 MB:

```
datasets/enterprise/corpus.jsonl           141 MB
pipeline3_round3/index_v2/edges_prov.json   55 MB
```

They are regenerable (`build/build_enterprise_corpus.py`, `build/canonicalize.py`) but that
takes hours. **Build on the machine that ran the benchmark** — it has them. The Dockerfile
checks for both and fails the build rather than shipping an image that starts cleanly and
then answers every question with nothing.

## Deploy

```bash
cd enterprise-graphrag

# 1. verify the image builds and runs locally first — this is the step that catches
#    a missing dependency, and it is much cheaper to find here
docker build -f deploy/Dockerfile -t graphrag-demo .
docker run --rm -p 8000:8000 --env-file .env graphrag-demo
#    then: curl localhost:8000/api/status   -> {"state":"ready"} within ~60s

# 2. first time only
fly launch --no-deploy --copy-config

# 3. secrets — NEVER in fly.toml, it is committed
fly secrets set \
  GOOGLE_API_KEY=... \
  TIGERGRAPH_HOST=https://<id>.i.tgcloud.io \
  TIGERGRAPH_SECRET=... \
  TIGERGRAPH_USERNAME=... \
  TIGERGRAPH_PASSWORD=... \
  DEMO_TOKEN=<a long random string>

# 4. ship
fly deploy --dockerfile deploy/Dockerfile
```

## The spend gate — read this before making the URL public

`/api/query` runs three LLM calls and a TigerGraph round trip: **~8,500 tokens and ~77
seconds per request, funded by whoever owns the Gemini key.** Unprotected on a public URL,
that is an anonymous endpoint on your account.

Two env vars, both inert unless set, so local development is unchanged:

| var | effect |
|---|---|
| `DEMO_TOKEN` | `/api/query` requires a matching `X-Demo-Token` header or `?token=`. Compared with `hmac.compare_digest`, so it does not leak length or prefix through timing. |
| `DEMO_RATE_PER_HOUR` | per-IP hourly cap. `fly.toml` sets 20. On refusal the page still browses the recorded runs, so hitting the limit degrades rather than breaks. |

This is **not authentication** — it is a spend limit with a shared password in front of it.
It stops crawlers and casual abuse. It does not stop someone who has the token from using
their quota, and it is not a substitute for SSO if this goes anywhere sensitive.

The rate limiter reads `X-Forwarded-For` and takes the first entry, which is correct behind
Fly, Railway and Render because they append rather than trust. On a platform that passes the
header through unmodified, the first entry is client-controlled and the limiter can be
bypassed by setting it — check before moving hosts.

## Pointing the static site at it

Once the service has a URL, the published page can call it instead of replaying. In
`demo/build_static.py`, replace the `STATIC_QUERY` shim with a `fetch` to
`https://<your-app>.fly.dev/api/query` carrying the token, and rebuild. Worth doing only if
you want live queries for visitors; the replay is honest, instant, free, and cannot break
during a talk.

## Known gaps

- **The Savanna workspace can suspend.** If it does, live queries fail while the static
  numbers stay fine. There is no keep-warm ping wired up.
- **No CI.** The image is built and pushed by hand from a machine holding the 196 MB of
  regenerable data. Automating it means putting that data in object storage first.
- **Single machine, no redundancy.** Fine for a demo, not for anything anyone depends on.
