import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

from glass.datasets.base import DatasetAdapter, EvaluationSample
from glass.datasets.registry import register

# Default paths relative to project root
_DEFAULT_CSV = "data/AA-LCR/AA-LCR_Dataset.csv"
_DEFAULT_EXTRACTED = "data/AA-LCR/AA-LCR_extracted-text/lcr"

# Prompt template matching the official AA-LCR specification:
# https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR
_PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


def _build_documents_text(docs: List[str]) -> str:
    """Format documents with numbered BEGIN/END markers."""
    return "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
        for i, doc in enumerate(docs)
    )


def _build_prompt(question: str, docs: List[str]) -> str:
    """Build the full prompt from question and ordered document texts."""
    documents_text = _build_documents_text(docs)
    return _PROMPT_TEMPLATE.format(documents_text=documents_text, question=question)


@register("aa_lcr")
class AALCRAdapter(DatasetAdapter):
    def __init__(self):
        self.samples: List[EvaluationSample] = []

    def load(self, dataset_config=None) -> None:
        """Load the AA-LCR dataset from a local folder.

        The dataset_folder config option should point to the extracted text
        directory (e.g. "data/AA-LCR/AA-LCR_extracted-text/lcr"). If not set,
        defaults to "data/AA-LCR/AA-LCR_extracted-text/lcr" relative to CWD.

        The CSV is expected at "data/AA-LCR/AA-LCR_Dataset.csv" relative to CWD,
        or alongside the dataset_folder's parent.
        """
        # Resolve dataset folder
        dataset_folder: Optional[str] = None
        if dataset_config and hasattr(dataset_config, "dataset_folder"):
            dataset_folder = dataset_config.dataset_folder

        if dataset_folder:
            dataset_folder = str(Path(dataset_folder).resolve())
        else:
            dataset_folder = str(Path(_DEFAULT_EXTRACTED).resolve())

        # Resolve CSV path: look next to the dataset folder's grandparent
        # e.g. data/AA-LCR/AA-LCR_extracted-text/lcr -> data/AA-LCR/AA-LCR_Dataset.csv
        csv_path = str(Path(dataset_folder).parent.parent / "AA-LCR_Dataset.csv")
        if not os.path.isfile(csv_path):
            # Fallback: try default path relative to CWD
            csv_path = str(Path(_DEFAULT_CSV).resolve())

        # Validate local data availability — exit early if missing
        if not os.path.isfile(csv_path):
            print(
                f"[GLASS] ERROR: AA-LCR dataset CSV not found at: {csv_path}\n"
                f"  Please download the dataset and set 'dataset.dataset_folder' in your config.\n"
                f"  See: https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR",
                file=sys.stderr,
            )
            raise FileNotFoundError(f"AA-LCR dataset CSV not found: {csv_path}")

        if not os.path.isdir(dataset_folder):
            print(
                f"[GLASS] ERROR: AA-LCR extracted text directory not found at: {dataset_folder}\n"
                f"  Please extract AA-LCR_extracted-text.zip and set 'dataset.dataset_folder' in your config.\n"
                f"  See: https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR",
                file=sys.stderr,
            )
            raise FileNotFoundError(f"AA-LCR extracted text directory not found: {dataset_folder}")

        # Load questions from CSV
        questions = self._load_questions(csv_path)

        # Build samples with full prompt
        for row in questions:
            doc_category = row["document_category"]
            doc_set_id = row["document_set_id"]
            filenames = row["data_source_filenames"]

            # Load document texts in the specified order
            docs = self._get_document_set(dataset_folder, doc_category, doc_set_id, filenames)

            # Build the full prompt with documents + question
            question_text = row["question"]
            context_prompt = _build_prompt(question_text, docs)

            # Parse input_tokens
            tokens = row.get("input_tokens")
            if tokens is not None:
                tokens = int(tokens)
            else:
                tokens = len(context_prompt) // 4

            sample = EvaluationSample(
                sample_id=str(row.get("question_id", "")),
                domain=doc_category,
                question=question_text,
                gold_answer=row.get("answer", ""),
                context_prompt=context_prompt,
                input_tokens=tokens,
                metadata={
                    "document_category": doc_category,
                    "document_set_id": doc_set_id,
                    "question_id": row.get("question_id", ""),
                    "data_source_filenames": filenames,
                    "data_source_urls": row.get("data_source_urls", ""),
                },
            )
            self.samples.append(sample)

    def get_samples(self) -> List[EvaluationSample]:
        return self.samples

    @staticmethod
    def _load_questions(csv_path: str) -> List[dict]:
        """Load AA-LCR questions from the local CSV file."""
        questions = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse data_source_filenames as ordered list (semicolon-separated)
                if "data_source_filenames" in row and isinstance(row["data_source_filenames"], str):
                    row["data_source_filenames"] = row["data_source_filenames"].split(";")

                questions.append(row)
        return questions

    @staticmethod
    def _get_document_set(
        dataset_folder: str,
        document_category: str,
        document_set_id: str,
        data_source_filenames: List[str],
    ) -> List[str]:
        """Load document texts in the order specified by data_source_filenames."""
        document_set_path = os.path.join(dataset_folder, document_category, document_set_id)

        document_texts = []
        for filename in data_source_filenames:
            document_path = os.path.join(document_set_path, filename)
            if not os.path.isfile(document_path):
                raise FileNotFoundError(
                    f"Document not found: {document_path}\n"
                    f"  Expected in document set: {document_category}/{document_set_id}"
                )
            with open(document_path, encoding="utf-8") as f:
                document_texts.append(f.read())
        return document_texts
