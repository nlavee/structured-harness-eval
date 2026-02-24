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
    def is_batchable(self) -> bool:
        return True

    def compute(self, output: RawOutput, sample: EvaluationSample, model_type: Optional[str] = None, **kwargs) -> Optional[float]:
        """
        Computes BERTScore F1 for a single sample.
        """
        results = self.compute_batch([output], [sample], model_type=model_type, **kwargs)
        return results[0]

    def compute_batch(self, outputs: list[RawOutput], samples: list[EvaluationSample], model_type: Optional[str] = None, **kwargs) -> list[Optional[float]]:
        """
        Computes BERTScore F1 for a batch of samples.
        Incurs model overhead only once.
        """
        preds = []
        golds = []
        valid_indices = []
        results = [None] * len(outputs)

        for i, (output, sample) in enumerate(zip(outputs, samples)):
            if output.error_type is not None:
                continue
            
            gold = sample.gold_answer
            pred = output.output

            if not gold or not gold.strip():
                continue

            if not pred or not pred.strip():
                results[i] = 0.0
                continue

            preds.append(pred)
            golds.append(gold)
            valid_indices.append(i)

        if not preds:
            return results

        # Only set lang="en" if we fall back to the default model to avoid warnings/conflicts
        use_default_lang = model_type is None
        lang = "en" if use_default_lang else None

        try:
            # P, R, F1 tensors returned
            _, _, f1_tensor = bert_score.score(
                preds,
                golds,
                model_type=model_type,
                lang=lang,
                verbose=False
            )
            
            f1_scores = f1_tensor.tolist()
            for idx, score in zip(valid_indices, f1_scores):
                results[idx] = float(score)
            
        except Exception as e:
            logger.error(f"BERTScore batch computation failed: {e}")
            # Keep existing None values in results
            
        return results
