import re

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

# Compiled once at import time for performance
_REFUSAL_PATTERNS = re.compile(
    r"|".join([
        r"I cannot answer",
        r"I can't answer",
        r"I am unable to answer",
        r"I don't have access",
        r"As an AI language model",
        r"I cannot provide",
        r"I am sorry",
        r"I'm sorry",
        r"I apologize",
        r"As a language model",
        r"I am an AI",
    ]),
    re.IGNORECASE,
)


@register("refusal_rate")
class RefusalMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        # Explicit refusal error type from inference phase
        if output.error_type == "refusal":
            return 1.0

        # For non-refusal errors (crash, timeout, api_error, malformed),
        # we cannot meaningfully assess refusal from the output text.
        if output.error_type:
            return None

        if _REFUSAL_PATTERNS.search(output.output):
            return 1.0

        return 0.0
