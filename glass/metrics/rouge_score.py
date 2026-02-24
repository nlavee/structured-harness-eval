import collections
from typing import Optional, Any

from rouge_score import rouge_scorer

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

# Fallback Score type matches rouge_scorer
Score = collections.namedtuple('Score', ['precision', 'recall', 'fmeasure'])

# Module-level cached scorer to avoid re-instantiating for every sample
_ROUGE_SCORER = None

def get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return _ROUGE_SCORER


class BaseRougeScoreMetric(BaseMetric):
    @property
    def category(self) -> str:
        return "correctness"

    def _compute_rouge(self, output: RawOutput, sample: EvaluationSample) -> Any:
        if output.error_type is not None:
            return None

        gold = sample.gold_answer
        pred = output.output

        # Undefined if gold is empty
        if not gold or not gold.strip():
            return None

        # Zero if prediction is empty
        if not pred or not pred.strip():
            return Score(precision=0.0, recall=0.0, fmeasure=0.0)

        scorer = get_rouge_scorer()
        scores = scorer.score(gold, pred)
        return scores['rougeL']


@register("rouge_score_f1")
class RougeScoreF1Metric(BaseRougeScoreMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> Optional[float]:
        score = self._compute_rouge(output, sample)
        return score.fmeasure if score is not None else None


@register("rouge_score_recall")
class RougeScoreRecallMetric(BaseRougeScoreMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> Optional[float]:
        score = self._compute_rouge(output, sample)
        return score.recall if score is not None else None
