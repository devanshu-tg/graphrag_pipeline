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

# CPU-only torch FIRST, and the right way to get it depends on the architecture:
#
#   x86_64   PyPI's default torch is the CUDA build — ~2 GB of GPU libraries in an image that
#            will only ever run on a CPU. The /whl/cpu index avoids that.
#   aarch64  there is no CUDA build for generic ARM, so PyPI's wheel is ALREADY CPU-only.
#            The /whl/cpu index carries no aarch64 wheels, so forcing it here fails the build
#            outright. Verified against the PyPI API: torch 2.13.0 publishes 12 aarch64 wheels
#            (manylinux_2_28_aarch64) and 6 x86_64.
#
# Oracle's Always Free tier is ARM, so both branches are live paths, not hypotheticals.
RUN ARCH="$(uname -m)"; \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        echo "aarch64 -> stock PyPI torch (already CPU-only)"; \
        pip install --no-cache-dir torch; \
    else \
        echo "$ARCH -> torch from the CPU index"; \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

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
# this is the check that catches a missing dependency here instead of on the first query —
# and it has already earned its keep once, catching tiktoken (imported by metrics.py, which
# a grep of the obvious files missed).
#
# pipeline3 is loaded by importlib at warm() time with an explicit path, so a plain
# `import pipeline3_round3.pipeline` does not exercise the code path the server uses. Load it
# the same way the server does, and import schema_router from inside that package, or this
# check passes while the only pipeline that matters is still broken.
RUN python -c "\
import sys, importlib.util; \
sys.path.insert(0, '.'); sys.path.insert(0, 'pipeline3_round3'); \
import pipeline1_llm_only, pipeline2_basic_rag, retrieval_common, metrics, llm_client, config; \
import schema_router; \
from embeddings.embedder import get_embedder; \
spec = importlib.util.spec_from_file_location('p3', 'pipeline3_round3/pipeline.py'); \
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
print('runtime imports OK — all three pipelines')"

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
