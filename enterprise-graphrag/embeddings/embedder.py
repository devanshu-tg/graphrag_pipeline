"""Local sentence-transformer embeddings, LangChain-compatible."""

from __future__ import annotations
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

import config


class LocalEmbedder(Embeddings):
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        print(f"[embedder] Loading {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

    # BAAI ships BGE with an instruction prefix on the QUERY side only (passages are embedded
    # bare). Retrieving without it is simply using the model wrong, and it costs real recall:
    # measured on the 20 multi-hop questions, answer coverage at k=10 went 0.307 -> 0.340.
    # Applied here so EVERY arm benefits equally — this is a retrieval fix, not a thumb on
    # the scale for one pipeline.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(self.QUERY_PREFIX + text,
                                 normalize_embeddings=True).tolist()


_embedder_instance: LocalEmbedder | None = None


def get_embedder() -> LocalEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = LocalEmbedder()
    return _embedder_instance
