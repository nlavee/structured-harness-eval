from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("gemini")
class GeminiSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        # Pass prompt via stdin to avoid ARG_MAX limits with large prompts (~400KB+).
        command = self.config.command or ["gemini", "--output-format", "text"]
        return self._run_command(command, sample.context_prompt, sample.sample_id)
