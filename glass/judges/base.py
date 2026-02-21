from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from pydantic import BaseModel


class EvalResult(BaseModel):
    sample_id: str
    system_name: str
    domain: str
    judge_model: Optional[str] = None  # "provider/model" used for judging
    metrics: Dict[str, Optional[float]]
    judge_outputs: Dict[str, str]
    human_label: Optional[int] = None


class Judge(ABC):
    @abstractmethod
    def evaluate_correctness(self, question: str, gold_answer: str, prediction: str) -> Tuple[float, str]:
        """
        Evaluate correctness of the prediction.
        Returns: (score, explanation)
        Score should be 1.0 for Correct, 0.0 for Incorrect.
        """
        pass

    @abstractmethod
    def evaluate_hallucination(self, prediction: str, context: str) -> Tuple[float, str]:
        """
        Evaluate hallucination rate.
        Returns: (rate, details)
        Rate is the fraction of sentences that are contradicted or unverified.
        """
        pass
