from typing import List

import datasets

from glass.datasets.base import DatasetAdapter, EvaluationSample
from glass.datasets.registry import register


@register("aa_lcr")
class AALCRAdapter(DatasetAdapter):
    def __init__(self):
        self.samples: List[EvaluationSample] = []

    def load(self) -> None:
        try:
            # Try loading 'test' split first, then 'train'
            try:
                ds = datasets.load_dataset("ArtificialAnalysis/AA-LCR", split="test")
            except Exception:
                ds = datasets.load_dataset("ArtificialAnalysis/AA-LCR", split="train")
        except Exception as e:
            raise RuntimeError(f"Failed to load AA-LCR dataset: {e}")

        for row in ds:
            # Mapping logic based on actual schema:
            # {'Unnamed: 0': 0, 'document_category': 'Academia', 'document_set_id': 'ac_markets', 'question_id': 1, 'question': '...', 'answer': '...', ...}
            context = row.get("context") or row.get("text") or row.get("question") or ""

            # Estimate tokens if not provided
            tokens = row.get("input_tokens")
            if tokens is None:
                tokens = len(context) // 4

            sample = EvaluationSample(
                sample_id=str(row.get("question_id", row.get("id", ""))),
                domain=row.get("document_category", "General"),
                question=row.get("question", ""),
                gold_answer=row.get("answer", ""),
                context_prompt=context,
                input_tokens=int(tokens),
                metadata=row,
            )
            self.samples.append(sample)

    def get_samples(self) -> List[EvaluationSample]:
        return self.samples
