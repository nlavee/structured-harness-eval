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
        result = self._run_command(command, sample.context_prompt, sample.sample_id)

        # Capture stderr as CoT when not quiet (Gemini logs thinking to stderr)
        if not self.config.quiet and result.stderr and not result.error_type:
            result.chain_of_thought = result.stderr

        return result
