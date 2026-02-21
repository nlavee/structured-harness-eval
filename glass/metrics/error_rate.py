from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput


@register("error_rate")
class ErrorRateMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample) -> float:
        return 1.0 if output.error_type or output.exit_code != 0 else 0.0
