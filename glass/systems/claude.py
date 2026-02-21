from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("claude")
class ClaudeSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        if self.config.command:
            command = self.config.command
        else:
            command = ["claude", "--print", "--output-format", "text"]
            if self.config.model:
                command += ["--model", self.config.model]
        return self._run_command(command, sample.context_prompt, sample.sample_id)
