# Hugging Face Spaces variant of deploy/Dockerfile.
#
# Three things differ from the Fly image, and all three are Spaces requirements rather than
# preferences:
#
#   1. PORT 7860. Spaces routes to that port unless `app_port` in README.md says otherwise.
#   2. NON-ROOT. Spaces runs the container as uid 1000, not root. Everything the process
#      writes to — the HF cache above all — must be owned by that user or the model load
#      fails with a permission error 60 seconds into boot, which is a miserable way to find
#      out. Ownership is set before the switch, not after.
#   3. NO .env. Secrets come from the Space's own settings and arrive as environment
#      variables. Nothing sensitive is baked into this image.
#
# The Space repo must contain the two large data files (git-lfs):
#   datasets/enterprise/corpus.jsonl           141 MB
#   pipeline3_round3/index_v2/edges_prov.json   55 MB
# The build fails loudly below if either is missing — see deploy/hf-space/README-deploy.md.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces' user. Created before anything is installed so the cache layers land with the right
# ownership rather than needing a recursive chown of a 3 GB image afterwards.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# CPU-only torch first — the default index resolves the CUDA build and drags ~2 GB of GPU
# libraries into an image that will only ever run on Spaces' free CPU tier.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY deploy/requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Bake the models in, as the user who will read them. A Space that downloads 220 MB on first
# request looks broken to whoever triggered the wake.
USER user
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('models cached')"

USER root
COPY --chown=user:user . /app
USER user

RUN test -s datasets/enterprise/corpus.jsonl \
    || (echo "MISSING corpus.jsonl — did git-lfs pull run? see deploy/hf-space/README-deploy.md" && exit 1)
RUN test -s pipeline3_round3/index_v2/edges_prov.json \
    || (echo "MISSING edges_prov.json — did git-lfs pull run?" && exit 1)

# Same check as the Fly image: import every runtime module, loading pipeline3 through
# importlib exactly as warm() does, so a missing dependency fails the build and not the first
# query. This already caught tiktoken once.
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
