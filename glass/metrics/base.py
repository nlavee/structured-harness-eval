from abc import ABC, abstractmethod
from typing import Optional

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, output: RawOutput, sample: EvaluationSample) -> Optional[float]:
        """Compute the metric score for the given raw output and sample."""
        pass
