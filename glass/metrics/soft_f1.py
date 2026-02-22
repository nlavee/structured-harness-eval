import collections

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.metrics.utils import normalize_answer
from glass.systems.base import RawOutput


@register("soft_f1")
class SoftF1Metric(BaseMetric):
    @property
    def category(self) -> str:
        return "correctness"

    def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
        if output.error_type:
            return None

        pred_tokens = normalize_answer(output.output).split()
        gold_tokens = normalize_answer(sample.gold_answer).split()

        if not gold_tokens:
            return None  # Undefined: gold answer is empty
            
        if not pred_tokens:
            return 0.0

        common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        
        return (2 * precision * recall) / (precision + recall)
