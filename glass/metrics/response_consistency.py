import re
from collections import Counter

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

# Patterns to extract key entities from text
_NUMBER_PATTERN = re.compile(r"\b\d[\d,.]*\b")  # "42", "1,234", "3.14"
_PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")  # "New York", "Paris"


def _extract_entities(text: str) -> list[str]:
    """Extract numbers and proper nouns from text."""
    numbers = _NUMBER_PATTERN.findall(text)
    proper_nouns = _PROPER_NOUN_PATTERN.findall(text)
    return numbers + proper_nouns


@register("response_consistency")
class ResponseConsistencyMetric(BaseMetric):
    """Measures internal consistency of entities across the response.

    Splits the response into two halves and compares entity mentions.
    Entities that appear in both halves are "consistent"; entities that
    appear in only one half are "inconsistent".

    Score = consistent_entities / total_unique_entities.

    A score of 1.0 means every entity mentioned in one half also appears
    in the other — strong internal consistency.  A low score may indicate
    the model is contradicting itself or drifting topic.

    Range: [0.0, 1.0].  Returns ``None`` on SUT error.
    Returns 1.0 if no entities are detected (vacuously consistent).

    **Research rationale**: Hallucinating systems often contradict themselves
    within a single response.  This deterministic metric complements the
    judge-based ``hallucination_rate``.
    """

    @property
    def category(self) -> str:
        return "behavioral"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None

        text = output.output.strip()
        if not text:
            return None  # No output to assess

        # Split into two halves at the midpoint
        midpoint = len(text) // 2
        first_half = text[:midpoint]
        second_half = text[midpoint:]

        entities_first = set(_extract_entities(first_half))
        entities_second = set(_extract_entities(second_half))

        all_entities = entities_first | entities_second
        if not all_entities:
            return 1.0  # No entities detected → vacuously consistent

        consistent = entities_first & entities_second
        return len(consistent) / len(all_entities)
