from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("gemini")
class GeminiSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        # Pass prompt via stdin to avoid ARG_MAX limits with large prompts (~400KB+).
        if self.config.command:
            command = self.config.command
        else:
            command = ["gemini", "--output-format", "text"]
            if self.config.model:
                command += ["--model", self.config.model]
        return self._run_command(command, sample.context_prompt, sample.sample_id)
