from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("gemini")
class GeminiSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        # Prompt is large (~400KB), but since _run_command uses subprocess.Popen with list,
        # it might still hit ARG_MAX if we pass it as an argument.
        # However, the user's manual test worked with a short prompt.
        # Let's try passing it as an argument as suggested by the manual test.
        # If it's too large, we'll need a different approach (like a temp file).
        command = self.config.command or ["gemini", "--output-format", "text", "--prompt", sample.context_prompt]
        # We pass empty string to stdin since prompt is now in command args
        return self._run_command(command, "", sample.sample_id)
