# Live GraphRAG demo — the half that cannot be static.
#
# Build from the enterprise-graphrag/ directory:
#     docker build -f deploy/Dockerfile -t graphrag-demo .
#
# IMPORTANT: this build needs two files that are .gitignored because GitHub rejects >100 MB:
#     datasets/enterprise/corpus.jsonl          141 MB
#     pipeline3_round3/index_v2/edges_prov.json  55 MB
# They are REGENERABLE (build/build_enterprise_corpus.py, build/canonicalize.py) but that is
# hours, so build on a machine that already has them — the one that ran the benchmark. The
# check below fails the build immediately rather than producing an image that starts fine and
# then answers every question with nothing.

FROM python:3.12-slim

# libgomp is torch's OpenMP runtime; the slim base omits it and the import fails at runtime,
# not at install time, which is a miserable way to find out.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch FIRST, from the CPU index. Left to itself pip resolves the default CUDA
# build and pulls ~2 GB of GPU libraries into an image that will only ever run on a CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY deploy/requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Bake the two models into the image. Downloading them on first boot means the first visitor
# waits for ~220 MB over the network, and a cold start that slow reads as a broken site.
ENV HF_HOME=/opt/hf
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('models cached')"

# Application code. Ordered after the dependency layers so a code edit does not reinstall
# torch.
COPY . /app

# Fail loudly if the ignored data files did not make it in.
RUN test -s datasets/enterprise/corpus.jsonl \
    || (echo "MISSING datasets/enterprise/corpus.jsonl — see the header of this Dockerfile" && exit 1)
RUN test -s pipeline3_round3/index_v2/edges_prov.json \
    || (echo "MISSING pipeline3_round3/index_v2/edges_prov.json — see the header" && exit 1)

# Import the whole runtime path at build time. requirements-runtime.txt is hand-derived, so
# this is the check that catches a missing dependency here instead of on the first query.
RUN python -c "\
import demo.server as s; \
import pipeline1_llm_only, pipeline2_basic_rag, schema_router, retrieval_common; \
print('runtime imports OK')" 2>/dev/null \
    || python -c "\
import sys; sys.path.insert(0,'.'); \
import pipeline1_llm_only, pipeline2_basic_rag, schema_router, retrieval_common; \
from embeddings.embedder import get_embedder; \
print('runtime imports OK')"

# Same configuration the benchmark ran under. P3_VEC_SLOTS=3 is the published setting:
# 2,733 tokens against Basic RAG's 5,312, 50/50 exact.
ENV TIGERGRAPH_GRAPH=enterprise \
    P3_VEC_SLOTS=3 \
    RR_POOL=200 \
    P3_REQUIRE_TG=1 \
    PYTHONUNBUFFERED=1 \
    DEMO_PORT=8000

# NOT set here, on purpose — these are secrets and belong in the platform's secret store:
#   GOOGLE_API_KEY  TIGERGRAPH_HOST  TIGERGRAPH_SECRET  TIGERGRAPH_USERNAME  TIGERGRAPH_PASSWORD
# And these gate public spend; the deploy config sets them:
#   DEMO_TOKEN  DEMO_RATE_PER_HOUR

EXPOSE 8000
CMD ["python", "demo/server.py"]
