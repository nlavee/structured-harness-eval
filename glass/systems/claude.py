import json
import logging

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register

logger = logging.getLogger(__name__)


@register("claude")
class ClaudeSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        if self.config.command:
            command = self.config.command
        else:
            output_format = "json" if not self.config.quiet else "text"
            command = ["claude", "--print", "--output-format", output_format, "--max-turns", "1"]
            if self.config.model:
                command += ["--model", self.config.model]
        result = self._run_command(command, sample.context_prompt, sample.sample_id)

        # When using json format, parse to extract CoT and clean output
        if not self.config.quiet and not self.config.command and result.output:
            result = self._parse_json_output(result)

        return result

    def _parse_json_output(self, result: RawOutput) -> RawOutput:
        """Parse JSON output to separate CoT from final answer."""
        try:
            data = json.loads(result.output)
            if isinstance(data, dict):
                result.output = data.get("result", result.output)
                # Capture any thinking/reasoning blocks
                if "thinking" in data:
                    result.chain_of_thought = data["thinking"]
        except (json.JSONDecodeError, KeyError):
            logger.debug("Could not parse Claude JSON output, keeping raw output")
        return result
