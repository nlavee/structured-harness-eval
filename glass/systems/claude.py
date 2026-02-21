from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("claude")
class ClaudeSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        command = self.config.command or [
            "claude",
            "--print",
            "--output-format",
            "text",
        ]
        return self._run_command(command, sample.context_prompt, sample.sample_id)
