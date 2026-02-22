import re

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

# Reasoning marker categories, based on discourse analysis literature.
# Each pattern is a word-boundary-delimited regex for reliable detection.

# Causal connectives indicate explicit reasoning chains
_CAUSAL_PATTERNS = re.compile(
    r"\b(?:because|therefore|thus|hence|consequently|as a result|"
    r"this (?:means|implies|suggests|indicates)|"
    r"due to|leads to|it follows)\b",
    re.IGNORECASE,
)

# Enumeration patterns indicate structured multi-step reasoning
_ENUMERATION_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]\s|[-•]\s|(?:first|second|third|fourth|fifth|finally|additionally|moreover|furthermore)[,:]?\s)",
    re.IGNORECASE | re.MULTILINE,
)

# Evidence markers indicate grounding claims in source material
_EVIDENCE_PATTERNS = re.compile(
    r"\b(?:according to|based on|the (?:document|text|source|report|data) (?:states?|shows?|indicates?|mentions?|describes?)|"
    r"as (?:stated|mentioned|described|noted) in|"
    r"evidence (?:shows?|suggests?|indicates?))\b",
    re.IGNORECASE,
)

# Normalization constant: a "deeply reasoned" response of ~200 words
# typically has ~8-12 markers.  We use 10 as the ceiling.
_MAX_EXPECTED_MARKERS = 10


@register("reasoning_depth")
class ReasoningDepthMetric(BaseMetric):
    """Measures structural depth of reasoning via linguistic markers.

    Counts three categories of reasoning markers:
    1. **Causal connectives**: "because", "therefore", "as a result"
    2. **Enumeration**: numbered lists, bullet points, ordinal transitions
    3. **Evidence grounding**: "according to", "the document states"

    Returns ``min(total_markers / 10, 1.0)`` — a score of 1.0 means the
    response exhibits at least 10 structural reasoning markers.

    Range: [0.0, 1.0].  Returns ``None`` on SUT error or empty output.

    **Research rationale**: Structured execution harnesses may induce more
    explicit reasoning chains due to CoT scaffolding.  This metric captures
    that behavioral signal.
    """

    @property
    def category(self) -> str:
        return "behavioral"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None

        text = output.output.strip()
        if not text:
            return 0.0

        causal = len(_CAUSAL_PATTERNS.findall(text))
        enumeration = len(_ENUMERATION_PATTERNS.findall(text))
        evidence = len(_EVIDENCE_PATTERNS.findall(text))

        total = causal + enumeration + evidence
        return min(total / _MAX_EXPECTED_MARKERS, 1.0)
