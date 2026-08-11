"""BERTScore — a judge-free accuracy control alongside the deterministic set F1.

Why both. The set score is exact and reproducible (gold answers are name sets computed from
the graph), but it is unforgiving: "eng-oncall, eng-platform" vs "eng-oncall and eng-platform"
scores the same, yet a near-miss and a total miss both score 0 on the missing item. BERTScore
gives partial credit for semantic overlap, and on the SEC run it was the metric that did NOT
flip between passes — the LLM judge flipped a question over identical answers.

Only rescale_with_baseline=False is used: the baseline rescaling is tuned for prose, and these
answers are comma-separated name lists.
"""
from __future__ import annotations

import os


class AccuracyEvaluator:
    """Thin wrapper over bert_score with the model pinned and loaded once."""

    def __init__(self, model: str | None = None, lang: str = "en"):
        self.model = model or os.getenv("BERTSCORE_MODEL", "roberta-large")
        self.lang = lang
        self._scorer = None

    def _get(self):
        if self._scorer is None:
            from bert_score import BERTScorer
            self._scorer = BERTScorer(model_type=self.model, lang=self.lang,
                                      rescale_with_baseline=False, idf=False)
        return self._scorer

    def bertscore(self, predictions: list[str], references: list[str]) -> list[float]:
        """F1 per pair. Empty predictions are dropped by the caller, not scored as 0 here —
        an empty string against a reference produces a meaningless similarity, not a zero."""
        if not predictions:
            return []
        refs = [r if isinstance(r, str) else ", ".join(r) for r in references]
        _, _, f1 = self._get().score(predictions, refs, verbose=False, batch_size=16)
        return [round(float(v), 4) for v in f1]
