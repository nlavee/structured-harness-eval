import re

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

# Pre-compiled patterns for linguistic confidence/hedging markers.
# Based on established hedging/boosting taxonomies in computational
# linguistics (Hyland 2005, "Metadiscourse").
_CONFIDENT_PATTERNS = re.compile(
    r"|".join([
        r"\bclearly\b",
        r"\bcertainly\b",
        r"\bdefinitely\b",
        r"\bundoubtedly\b",
        r"\bwithout doubt\b",
        r"\bthe answer is\b",
        r"\bspecifically\b",
        r"\bprecisely\b",
        r"\bin fact\b",
        r"\bconfirm(?:s|ed)?\b",
        r"\bevidence shows\b",
        r"\bdemonstrates?\b",
    ]),
    re.IGNORECASE,
)

_HEDGING_PATTERNS = re.compile(
    r"|".join([
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bI'm not sure\b",
        r"\bI am not sure\b",
        r"\bI think\b",
        r"\bI believe\b",
        r"\bmay be\b",
        r"\bprobably\b",
        r"\bseems? to\b",
        r"\bappears? to\b",
        r"\bit is unclear\b",
        r"\buncertain\b",
    ]),
    re.IGNORECASE,
)


@register("confidence_score")
class ConfidenceScoreMetric(BaseMetric):
    """Ratio of confident vs hedging language markers.

    Computes: confident_count / (confident_count + hedging_count).
    Returns 0.5 (neutral) if no markers found.  Range: [0.0, 1.0].

    A higher score indicates more authoritative language.  This is a
    behavioral signal — it does NOT measure whether the answer is correct.
    """

    @property
    def category(self) -> str:
        return "behavioral"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None

        text = output.output
        confident_count = len(_CONFIDENT_PATTERNS.findall(text))
        hedging_count = len(_HEDGING_PATTERNS.findall(text))

        total = confident_count + hedging_count
        if total == 0:
            return 0.5  # Neutral: no confidence/hedging markers detected

        return confident_count / total
