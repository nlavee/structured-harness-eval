from glass.config.schema import JudgeConfig
from glass.judges.base import Judge
from glass.judges.llm import LLMJudge
from glass.judges.strategies.base import JudgeStrategy


class RotationStrategy(JudgeStrategy):
    def __init__(self, config: JudgeConfig):
        self.config = config

    def assign_judge(self, system_name: str) -> Judge:
        if self.config.strategy == "fixed":
            if not self.config.fixed:
                raise ValueError("Strategy is 'fixed' but no fixed judge config provided.")
            return LLMJudge(self.config.fixed.provider, self.config.fixed.model)

        # Rotation logic
        name = system_name.lower()
        family = "openai"  # Default fallback?

        if "claude" in name:
            family = "anthropic"
        elif "gemini" in name:
            family = "google"
        elif "codex" in name or "gpt" in name or "openai" in name:
            family = "openai"

        # Get rotation config for this family
        # config.rotation has attributes anthropic, google, openai (JudgeProviderConfig)
        if not self.config.rotation:
            raise ValueError("Strategy is 'rotation' but no rotation config provided.")

        provider_config = getattr(self.config.rotation, family, None)

        if not provider_config:
            # If we can't map family, maybe use fixed?
            if self.config.fixed:
                return LLMJudge(self.config.fixed.provider, self.config.fixed.model)
            raise ValueError(f"Could not determine judge rotation for system '{system_name}' (family: {family})")

        return LLMJudge(provider_config.provider, provider_config.model)
