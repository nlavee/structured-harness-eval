import re

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

# Compiled once at import time for performance
_CITATION_PATTERNS = re.compile(
    r"|".join([
        r"\[doc(?:ument)?\s*\d+\]",   # [doc 1], [Document 3]
        r"source:\s*\d+",             # source: 1
        r"\[\d{1,3}\]",               # [1], [2], [42] — cap at 3 digits to avoid [2023] false positives
    ]),
    re.IGNORECASE,
)


@register("citation_presence")
class CitationPresenceMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None

        if _CITATION_PATTERNS.search(output.output):
            return 1.0
        return 0.0
