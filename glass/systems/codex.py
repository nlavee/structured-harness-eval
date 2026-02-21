from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("codex")
class CodexSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        if self.config.command:
            command = self.config.command
        else:
            model = self.config.model or "gpt-5"
            command = ["codex", "exec", "--model", model, "--ask-for-approval", "never"]
        return self._run_command(command, sample.context_prompt, sample.sample_id)
