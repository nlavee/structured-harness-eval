from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("gemini")
class GeminiSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        command = self.config.command or ["gemini", "-p"]
        return self._run_command(command, sample.context_prompt, sample.sample_id)
