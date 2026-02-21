from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput


@register("latency_s")
class LatencyMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample) -> float:
        return output.latency_s
