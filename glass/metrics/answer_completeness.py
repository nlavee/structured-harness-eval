import re

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput


@register("answer_completeness")
class AnswerCompletenessMetric(BaseMetric):
    """Measures how many sub-questions in a compound question are addressed.

    Heuristic: count question marks in ``sample.question`` as sub-question
    count.  Count distinct answer sentences as coverage.  Returns
    ``min(answer_sentences / question_parts, 1.0)``.

    For single-question prompts (1 question mark), this reduces to a binary:
    did the model produce at least one sentence?  For multi-part questions,
    it captures whether the model addressed each part.

    Range: [0.0, 1.0].  Returns ``None`` on SUT error.
    """

    @property
    def category(self) -> str:
        return "correctness"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None

        # Count sub-questions (at least 1 even if no explicit question mark)
        question_parts = max(sample.question.count("?"), 1)

        # Count answer sentences using a simple but robust splitter
        # (period, exclamation, question mark followed by space or end)
        text = output.output.strip()
        if not text:
            return 0.0

        sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()]
        answer_sentences = max(len(sentences), 1) if text else 0

        return min(answer_sentences / question_parts, 1.0)
