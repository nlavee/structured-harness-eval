from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput


@register("verbosity")
class VerbosityMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample) -> float:
        if output.error_type:
            return None
        gold_len = len(sample.gold_answer)
        if gold_len == 0:
            return None  # Undefined: can't compute ratio against empty gold answer
        return len(output.output) / gold_len
