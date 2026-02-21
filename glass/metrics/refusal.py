import re

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput


@register("refusal_rate")
class RefusalMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample) -> float:
        # Pattern based refusal
        if output.error_type == "refusal":  # Explicit error
            return 1.0

        patterns = [
            r"I cannot answer",
            r"I can't answer",
            r"I am unable to answer",
            r"I don't have access",
            r"As an AI language model",
            r"I cannot provide",
        ]
        text = output.output
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return 1.0
        return 0.0
