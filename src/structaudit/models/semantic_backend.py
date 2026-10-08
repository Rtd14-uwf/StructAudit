"""
Semantic model backend
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from rapidfuzz import fuzz, utils

from structaudit.core import SemanticCallLog

_SUPPORTED_THRESHOLD = 60
_NOT_SUPPORTED_THRESHOLD = 40
_LABEL_EQUIVALENT_THRESHOLD = 90
_LABEL_DIFFERENT_THRESHOLD = 60

class SemanticBackend(ABC):
    """One call log list, shared by every concrete backend, so callers can
    inspect what was actually asked/answered regardless of which backend is in use.
    """

    def __init__(self) -> None:
        self.call_log: list[SemanticCallLog] = []

    @abstractmethod
    def _classify_impl(self, task: str, payload: dict) -> tuple[dict, dict]:
        """Return (raw_output, parsed_output) for one call. Subclasses
        implement this; classify() wraps it with timing and logging.
        """
        ...

    @property
    @abstractmethod
    def model_backend(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    def classify(self, task: str, payload: dict) -> dict:
        start = time.monotonic()
        raw_output, parsed_output = self._classify_impl(task, payload)
        latency = time.monotonic() - start

        self.call_log.append(
            SemanticCallLog(
                task=task,
                model_backend=self.model_backend,
                model_name=self.model_name,
                input=payload,
                raw_output=raw_output,
                parsed_output=parsed_output,
                token_usage=None,  # only meaningful for an API backend; see ApiSemanticBackend when built
                latency=latency,
            )
        )
        return parsed_output


class LexicalOverlapBackend(SemanticBackend):
    """Local backend using rapidfuzz token-set overlap as a stand-in for a real NLI model."""

    @property
    def model_backend(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return "lexical-overlap-v1"

    def _classify_impl(self, task: str, payload: dict) -> tuple[dict, dict]:
        if task == "reference_topic_support":
            return self._reference_topic_support(payload)
        if task == ("schema_label_equivalence", "coverage_member_equivalence"):
            return self._label_equivalence(payload)
        raise ValueError(f"LexicalOverlapBackend does not support task {task!r}")

    def _reference_topic_support(self, payload: dict) -> tuple[dict, dict]:
        expected_topic = payload["expected_topic"]
        resolved_target_text = payload["resolved_target_text"]

        score = fuzz.token_set_ratio(expected_topic, resolved_target_text)

        if score >= _SUPPORTED_THRESHOLD:
            label = "SUPPORTED"
        elif score < _NOT_SUPPORTED_THRESHOLD:
            label = "NOT_SUPPORTED"
        else:
            label = "UNCERTAIN"

        raw_output = {"token_set_ratio": score}
        parsed_output = {"label": label, "score": score}
        return raw_output, parsed_output

    def _label_equivalence(self, payload: dict) -> tuple[dict, dict]:
        """Is expected_label equivalent to any of candidate_labels? Returns the
        best candidate and SUPPORTED / NOT_SUPPORTED / UNCERTAIN
        """
        expected_label = payload["expected_label"]
        candidate_labels = payload["candidate_labels"]

        scored = [
            (fuzz.token_sort_ratio(expected_label, label, processor=utils.default_process), label)
            for label in candidate_labels
        ]
        best_score, best_label = max(scored, default=(0.0, None))

        if best_score >= _LABEL_EQUIVALENT_THRESHOLD:
            label, confidence = "SUPPORTED", best_score / 100
        elif best_score < _LABEL_DIFFERENT_THRESHOLD:
            label, confidence = "NOT_SUPPORTED", (100 - best_score) / 100
        else:
            label, confidence = "UNCERTAIN", 0.0

        raw_output = {"best_token_sort_ratio": best_score, "best_candidate": best_label}
        parsed_output = {
            "label": label,
            "best_match": best_label if label == "SUPPORTED" else None,
            "score": best_score,
            "confidence": confidence,
        }

        return raw_output, parsed_output
