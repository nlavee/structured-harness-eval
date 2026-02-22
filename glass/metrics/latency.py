import logging

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

logger = logging.getLogger(__name__)


@register("latency_s")
class LatencyMetric(BaseMetric):
    """Wall-clock subprocess time (AP-9).

    Always returns the measured latency — even when the SUT errors or times
    out — because the wall-clock time is valid operational data regardless
    of output quality.
    """

    @property
    def category(self) -> str:
        return "operational"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            logger.debug(
                "Latency recorded with error_type='%s' for %s/%s: %.2fs",
                output.error_type,
                output.system_name,
                sample.sample_id,
                output.latency_s,
            )
        return output.latency_s
