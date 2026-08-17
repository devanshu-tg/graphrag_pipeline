# Hugging Face Spaces variant of deploy/Dockerfile.
#
# BUILD CONTEXT IS THE SPACE REPO ROOT, not enterprise-graphrag/. server.py resolves
# EVID = ROOT.parent/"_evidence" and MG_ROOT = ROOT.parent/"mastergraph-round3", so with the
# app at /app those become /_evidence and /mastergraph-round3. Building from inside
# enterprise-graphrag/ puts them out of context and the SEC tab silently loses its question
# list. The Space repo therefore mirrors the project layout and this file copies three trees
# to the three places the code expects.
#
# Differences from the Fly image, all Spaces requirements rather than preferences:
#   1. PORT 7860 — Spaces routes there unless app_port in README.md says otherwise.
#   2. NON-ROOT uid 1000. Ownership is set BEFORE the user switch; get that order wrong and
#      the model load dies on a permission error 60 seconds into boot.
#   3. NO .env — secrets arrive from the Space's own settings.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# CPU-only torch first — the default index resolves the CUDA build and drags ~2 GB of GPU
# libraries into an image that will only ever run on Spaces' free CPU tier.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY enterprise-graphrag/deploy/requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Bake the models in as the user that will read them. A Space that downloads 220 MB on the
# first request looks broken to whoever triggered the wake.
USER user
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('models cached')"

USER root
COPY --chown=user:user enterprise-graphrag /app
# The two siblings server.py expects one level above the app.
COPY --chown=user:user _evidence /_evidence
COPY --chown=user:user mastergraph-round3 /mastergraph-round3
USER user

RUN test -s datasets/enterprise/corpus.jsonl \
    || (echo "MISSING corpus.jsonl — see deploy/hf-space/README-deploy.md" && exit 1)
RUN test -s pipeline3_round3/index_v2/edges_prov.json \
    || (echo "MISSING edges_prov.json — see deploy/hf-space/README-deploy.md" && exit 1)
RUN test -s results/enterprise/eval_pub50.jsonl \
    || (echo "MISSING eval_pub50.jsonl — the dashboard tables read it" && exit 1)

# Import every runtime module, loading pipeline3 through importlib exactly as warm() does, so
# a missing dependency fails the build rather than the first query. This already caught
# tiktoken once.
RUN python -c "\
import sys, importlib.util; \
sys.path.insert(0, '.'); sys.path.insert(0, 'pipeline3_round3'); \
import pipeline1_llm_only, pipeline2_basic_rag, retrieval_common, metrics, llm_client, config; \
import schema_router; \
from embeddings.embedder import get_embedder; \
spec = importlib.util.spec_from_file_location('p3', 'pipeline3_round3/pipeline.py'); \
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
print('runtime imports OK — all three pipelines')"

ENV TIGERGRAPH_GRAPH=enterprise \
    P3_VEC_SLOTS=3 \
    RR_POOL=200 \
    P3_REQUIRE_TG=1 \
    PYTHONUNBUFFERED=1 \
    DEMO_PORT=7860 \
    DEMO_RATE_PER_HOUR=20

# Set in the Space's Settings -> Variables and secrets, never here:
#   GOOGLE_API_KEY  TIGERGRAPH_HOST  TIGERGRAPH_SECRET  TIGERGRAPH_USERNAME
#   TIGERGRAPH_PASSWORD  DEMO_TOKEN

EXPOSE 7860
CMD ["python", "demo/server.py"]
