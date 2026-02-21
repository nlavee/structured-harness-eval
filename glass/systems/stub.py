import time

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("stub")
class StubSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        time.sleep(0.1)
        command = self.config.command or ["stub"]
        return RawOutput(
            sample_id=sample.sample_id,
            system_name=self.config.name,
            command=command,
            prompt=sample.context_prompt,
            output=sample.gold_answer,  # Perfect score
            latency_s=0.1,
            exit_code=0,
            stderr="",
            error_type=None,
            timestamp="2023-01-01T00:00:00",
        )
