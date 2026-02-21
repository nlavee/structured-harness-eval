from importlib import resources

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput, SystemUnderTest
from glass.systems.registry import register


@register("structured_harness")
class StructuredHarnessSystem(SystemUnderTest):
    def generate(self, sample: EvaluationSample) -> RawOutput:
        # Resolve config json path
        config_name = self.config.harness_config or "single_turn_qa"

        # Try to find it in the package
        try:
            # Python 3.9+
            ref = resources.files("glass.systems.structured_configs") / f"{config_name}.json"
            with resources.as_file(ref) as p:
                config_path = str(p)
        except Exception:
            # Fallback for older python or if not installed as package
            config_path = f"glass/systems/structured_configs/{config_name}.json"

        command = [
            "structured-harness",
            config_path,
            "-",  # stdin
            "--model",
            self.config.model or "gpt-5",
            "--headless",
            "--autostart",
        ]

        return self._run_command(command, sample.context_prompt, sample.sample_id)
