from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class EvaluationSample(BaseModel):
    sample_id: str
    domain: str
    question: str
    gold_answer: str
    context_prompt: str
    input_tokens: int
    turn_type: Literal["single", "multi"] = "single"
    prior_turns: List[ConversationTurn] = []
    metadata: Dict[str, Any]


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self, dataset_config=None) -> None:
        """Load the dataset (download, extract, etc.)
        
        Args:
            dataset_config: Optional DatasetConfig with settings like dataset_folder.
        """
        pass

    @abstractmethod
    def get_samples(self) -> List[EvaluationSample]:
        """Return the list of evaluation samples"""
        pass
