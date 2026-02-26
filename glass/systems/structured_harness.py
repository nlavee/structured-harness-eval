from importlib import resources

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("structured_harness")
class StructuredHarnessSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        raise NotImplementedError("Structured Harness is a standin right now.")