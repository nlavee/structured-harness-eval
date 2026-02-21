from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.metrics.utils import normalize_answer
from glass.systems.base import RawOutput


@register("exact_match")
class ExactMatchMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample) -> float:
        if output.error_type:
            return None
        return 1.0 if normalize_answer(output.output) == normalize_answer(sample.gold_answer) else 0.0
