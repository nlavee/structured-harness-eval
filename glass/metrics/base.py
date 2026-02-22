from abc import ABC, abstractmethod
from typing import Optional

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput


class BaseMetric(ABC):
    """Base class for all GLASS metrics.

    Subclasses must implement ``compute()``.  Two optional properties control
    how the pipeline treats a metric:

    * ``requires_judge`` – if ``True``, the pipeline passes a ``judge``
      instance and ``judge_outputs`` dict via **kwargs.  Defaults to ``False``.
    * ``category`` – a human-readable grouping used for documentation and
      reporting.  One of ``"correctness"``, ``"behavioral"``, or
      ``"operational"``.
    """

    @property
    def requires_judge(self) -> bool:
        """Whether this metric needs a judge instance passed via kwargs."""
        return False

    @property
    def category(self) -> str:
        """Metric category: 'correctness', 'behavioral', or 'operational'."""
        return "behavioral"

    @abstractmethod
    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> Optional[float]:
        """Compute the metric score for the given raw output and sample.

        Returns:
            A float score, or ``None`` if the metric is undefined for this
            input (e.g. SUT error, empty gold answer).  Never return ``0.0``
            for an error condition (AP-3).
        """
        pass

