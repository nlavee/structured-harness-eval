import collections

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.metrics.utils import normalize_answer
from glass.systems.base import RawOutput


@register("soft_recall")
class SoftRecallMetric(BaseMetric):
    def compute(self, output: RawOutput, sample: EvaluationSample) -> float:
        if output.error_type:
            return None

        pred_tokens = normalize_answer(output.output).split()
        gold_tokens = normalize_answer(sample.gold_answer).split()

        if not gold_tokens:
            return None  # Undefined: gold answer is empty

        common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values())
        return num_same / len(gold_tokens)  # Recall = TP / (TP + FN) = overlap / gold_len
