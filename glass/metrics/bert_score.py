import logging
from typing import Optional

import bert_score

from glass.datasets.base import EvaluationSample
from glass.metrics.base import BaseMetric
from glass.metrics.registry import register
from glass.systems.base import RawOutput

logger = logging.getLogger(__name__)


@register("bert_score_f1")
class BertScoreF1Metric(BaseMetric):
    @property
    def category(self) -> str:
        return "correctness"

    def compute(self, output: RawOutput, sample: EvaluationSample, model_type: Optional[str] = None, **kwargs) -> Optional[float]:
        """
        Computes BERTScore F1.
        
        TODO: `bert_score` automatically batches, but GLASS processes samples one by one in Phase 2. 
        This adds slight per-sample BERT overhead and can be optimized in the future by batching metrics.
        """
        if output.error_type is not None:
            return None

        gold = sample.gold_answer
        pred = output.output

        # Undefined if gold is empty
        if not gold or not gold.strip():
            return None

        # Zero if prediction is empty
        if not pred or not pred.strip():
            return 0.0
            
        # Only set lang="en" if we fall back to the default model to avoid warnings/conflicts
        use_default_lang = model_type is None
        lang = "en" if use_default_lang else None

        try:
            # P, R, F1 tensors returned
            _, _, f1_tensor = bert_score.score(
                [pred],
                [gold],
                model_type=model_type,
                lang=lang,
                verbose=False
            )
            return float(f1_tensor.item())
        except Exception as e:
            logger.error(f"BERTScore computation failed for sample {sample.sample_id}: {e}")
            return None
