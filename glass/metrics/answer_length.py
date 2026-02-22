from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

@register("answer_length")
class AnswerLengthMetric(BaseMetric):
    @property
    def category(self) -> str:
        return "behavioral"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None
        return float(len(output.output.split()))
