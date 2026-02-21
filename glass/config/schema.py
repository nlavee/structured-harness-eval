from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    name: str
    run_id: Optional[str] = None
    seed: int


class DatasetConfig(BaseModel):
    name: str
    samples: Optional[int] = None
    domains: Optional[List[str]] = None


class SystemConfig(BaseModel):
    name: str
    type: str
    command: Optional[List[str]] = None
    model: Optional[str] = None
    harness_config: Optional[str] = None
    timeout_s: int = 600
    env: Dict[str, str] = Field(default_factory=dict)


class JudgeProviderConfig(BaseModel):
    provider: str
    model: str


class RotationConfig(BaseModel):
    anthropic: JudgeProviderConfig
    google: JudgeProviderConfig
    openai: JudgeProviderConfig


class JudgeTemplatesConfig(BaseModel):
    correctness: str
    hallucination: str


class JudgeConfig(BaseModel):
    strategy: Literal["rotation", "fixed"]
    fixed: Optional[JudgeProviderConfig] = None
    rotation: Optional[RotationConfig] = None
    templates: JudgeTemplatesConfig


class StatisticsConfig(BaseModel):
    bootstrap_resamples: int = 10000
    alpha: float = 0.05
    primary_test: str
    secondary_test: str


class OutputConfig(BaseModel):
    runs_dir: str = "./runs"
    log_level: str = "INFO"


class Config(BaseModel):
    experiment: ExperimentConfig
    dataset: DatasetConfig
    systems: List[SystemConfig]
    metrics: List[str]
    judges: JudgeConfig
    statistics: StatisticsConfig
    output: OutputConfig
