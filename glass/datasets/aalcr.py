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
            # Mapping logic - adjusting based on likely schema
            # Assuming 'context' or 'text' contains the full context
            context = row.get("context") or row.get("text") or ""

            # Estimate tokens if not provided
            tokens = row.get("input_tokens")
            if tokens is None:
                tokens = len(context) // 4

            sample = EvaluationSample(
                sample_id=str(row.get("id", row.get("sample_id", ""))),
                domain=row.get("domain", "General"),
                question=row.get("question", ""),
                gold_answer=row.get("answer", row.get("gold_answer", "")),
                context_prompt=context,
                input_tokens=int(tokens),
                metadata=row,
            )
            self.samples.append(sample)

    def get_samples(self) -> List[EvaluationSample]:
        return self.samples
