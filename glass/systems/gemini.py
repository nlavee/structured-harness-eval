from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register
import json
import logging

logger = logging.getLogger(__name__)


@register("gemini")
class GeminiSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        # Pass prompt via stdin to avoid ARG_MAX limits with large prompts (~400KB+).
        if self.config.command:
            command = self.config.command
        else:
            fmt = self.config.output_format or "text"
            command = ["gemini", "--output-format", fmt]
            if self.config.model:
                command += ["--model", self.config.model]
        result = self._run_command(command, sample.context_prompt, sample.sample_id)

        # Map stream-json if configured
        if getattr(self.config, "output_format", None) == "stream-json" and result.output:
            result = self._parse_stream_json(result)
        # Capture stderr as CoT when not quiet (Gemini logs thinking to stderr when not using stream-json)
        elif not self.config.quiet and result.stderr and not result.error_type:
            result.chain_of_thought = result.stderr

        return result

    def _parse_stream_json(self, result: RawOutput) -> RawOutput:
        """Parse streaming JSONL from Gemini CLI output."""
        full_output = ""
        full_reasoning = ""
        tool_calls = []

        try:
            for line in result.output.strip().splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)

                # Capture final answer
                if data.get("type") == "message" and data.get("role") == "assistant":
                    if "content" in data:
                        full_output += data["content"]
                    
                    if "thinking" in data and not self.config.quiet:
                         full_reasoning += data["thinking"]

                # Capture tool calls and results
                elif data.get("type") in ["tool_use", "tool_result"]:
                    tool_calls.append(data)

            if full_output:
                 result.output = full_output
            
            if full_reasoning:
                 result.chain_of_thought = full_reasoning
                 
            if tool_calls:
                 result.tool_calls = tool_calls

        except json.JSONDecodeError:
            logger.debug("Could not parse Gemini JSONL output, keeping raw output")
        
        return result
