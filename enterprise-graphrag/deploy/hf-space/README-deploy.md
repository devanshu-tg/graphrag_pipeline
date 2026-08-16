# Pushing the live service to Hugging Face Spaces

Free CPU tier: 2 vCPU, 16 GB RAM, 50 GB disk. This service needs 1,064 MB resident and a
~3 GB image, so it fits with a lot of room. No credit card.

Why Spaces rather than Fly: it is free and permanent. The trade is a cold wake of ~60 s after
48 h idle, which for a demo people open occasionally is a better deal than $5–10/month.

## What you need

An HF token with **write** access — https://huggingface.co/settings/tokens

## Steps

```bash
# 1. create the Space (Docker SDK) — via the UI or the CLI
pip install -U "huggingface_hub[cli]"
hf auth login                       # paste the write token
hf repo create graphrag-demo --repo-type space --space_sdk docker

# 2. assemble the Space repo. It needs the app, the two large data files, this Dockerfile
#    at the ROOT, and README.md at the ROOT (its YAML frontmatter is the Space config).
git clone https://huggingface.co/spaces/<you>/graphrag-demo hf-space && cd hf-space
git lfs install
git lfs track "*.jsonl" "edges_prov.json"

#    copy from the project — Dockerfile and README.md must land at the repo root
cp ../enterprise-graphrag/deploy/hf-space/Dockerfile .
cp ../enterprise-graphrag/deploy/hf-space/README.md .
rsync -a --exclude 'deploy' --exclude '__pycache__' --exclude '.env' \
      ../enterprise-graphrag/ ./
cp -r ../enterprise-graphrag/deploy/requirements-runtime.txt deploy/

#    the two gitignored files must be copied EXPLICITLY — the project ignores them for size
cp ../enterprise-graphrag/datasets/enterprise/corpus.jsonl datasets/enterprise/
cp ../enterprise-graphrag/pipeline3_round3/index_v2/edges_prov.json pipeline3_round3/index_v2/

git add -A && git commit -m "GraphRAG live service" && git push
```

## 3. Secrets — Space Settings → Variables and secrets

**Never commit these.** The Dockerfile deliberately bakes none of them.

| name | value |
|---|---|
| `GOOGLE_API_KEY` | Gemini key |
| `TIGERGRAPH_HOST` | `https://<id>.i.tgcloud.io` |
| `TIGERGRAPH_SECRET` | Savanna secret |
| `TIGERGRAPH_USERNAME` | usually `tigergraph` |
| `TIGERGRAPH_PASSWORD` | |
| `DEMO_TOKEN` | a long random string, if you want the query box gated |

`DEMO_RATE_PER_HOUR` is already set to 20 in the Dockerfile; override it here to change it.

## 4. Point the published page at it

Once the Space is live at `https://<you>-graphrag-demo.hf.space`, replace the `STATIC_QUERY`
shim in `demo/build_static.py` with a `fetch` to `<space-url>/api/query`, rebuild, redeploy.
Then graphrag-benchmark.netlify.app runs live queries against the Space.

Keep the replay as the fallback: if the Space is asleep, the first request takes ~60 s, and a
page that answers instantly from the recording while the backend wakes is strictly better
than a spinner.

## Things that will bite

- **`.env` must not be copied into the Space repo.** The rsync above excludes it. Check
  `git status` before the first push — a Space repo is public unless you made it private, and
  its git history is public with it.
- **`git-lfs` must be installed and tracking before you add the big files**, or the push is
  rejected. `git lfs track` writes `.gitattributes`; commit that too.
- **The build runs on HF's machines**, so the ~3 GB image is built there, not uploaded. First
  build takes 10–20 minutes, mostly torch.
- **The Space sleeps after 48 h idle** on the free tier and wakes on the next request. Cold
  wake is ~60 s: 141 MB corpus into a dict, 55 MB edge index, two models.
- **A public Space shows its files and its git history to anyone.** The corpus is fictional
  and the code is already on GitHub, so the only real risk is a leaked secret — see the first
  bullet. Make the Space private if you would rather not publish the corpus at all; private
  Spaces are free too, but then the page needs the token to reach it.
