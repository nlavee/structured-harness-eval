"""Finch (FinWorkBench) dataset adapter for GLASS.

Loads the FinWorkBench/Finch dataset from HuggingFace:
https://huggingface.co/datasets/FinWorkBench/Finch

Finch contains 172 enterprise accounting/finance workflow tasks spanning
9 categories (validation, calculation, data entry, structuring, etc.).
Each task has source files (.xlsx, .csv, .pdf) and reference outputs.

This adapter provides two interfaces:
  - get_samples() -> List[EvaluationSample]  (standard GLASS compatibility)
  - get_tasks()   -> List[FinchTask]          (rich task data for accounting pipeline)

Usage in config:
    dataset:
      name: "finch"
      dataset_folder: "data/Finch"   # optional, defaults to data/Finch
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from glass.datasets.base import DatasetAdapter, EvaluationSample
from glass.datasets.registry import register

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data/Finch"


class FinchReferenceOutput(BaseModel):
    """Reference output for a Finch task."""
    files: List[str] = []
    text: Optional[str] = None


class FinchTask(BaseModel):
    """A single Finch workflow task with full metadata.

    This is the rich representation used by the accounting pipeline.
    It preserves source file paths, reference outputs, task type taxonomy,
    and business domain — all needed for chain construction and evaluation.
    """
    task_id: str
    instruction: str
    source_file_paths: List[str]
    source_file_urls: List[str]
    reference_output_paths: List[str]
    reference_output_urls: List[str]
    reference_text: Optional[str] = None
    task_types: List[str]
    business_type: str
    task_constraints: Optional[str] = None
    metadata: Dict[str, Any] = {}


def _parse_task_types(raw: str) -> List[str]:
    """Parse task_type field which may be comma-separated or a single value."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_list_field(raw) -> List[str]:
    """Parse a field that may be a list, JSON string, or None."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                return [str(x) for x in parsed] if isinstance(parsed, list) else [raw]
            except json.JSONDecodeError:
                pass
        return [x.strip() for x in raw.split(";") if x.strip()]
    return [str(raw)]


def _parse_reference_outputs(raw) -> FinchReferenceOutput:
    """Parse the reference_outputs field which may be a dict or JSON string."""
    if raw is None:
        return FinchReferenceOutput()
    if isinstance(raw, dict):
        return FinchReferenceOutput(
            files=_parse_list_field(raw.get("files")),
            text=raw.get("text"),
        )
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return FinchReferenceOutput(
                        files=_parse_list_field(parsed.get("files")),
                        text=parsed.get("text"),
                    )
            except json.JSONDecodeError:
                pass
        return FinchReferenceOutput(text=raw if raw else None)
    return FinchReferenceOutput()


@register("finch")
class FinchAdapter(DatasetAdapter):
    """Dataset adapter for the FinWorkBench/Finch accounting benchmark.

    Loads Finch tasks from a local data directory. Run scripts/download_finch.py
    first to download and set up the data.

    Directory layout expected:
        data/Finch/
        ├── finch_tasks.json       # Task metadata (exported from HuggingFace)
        ├── source_files/          # Downloaded source files per task
        │   ├── 0/
        │   │   ├── 0_src_0.xlsx
        │   │   └── ...
        │   └── ...
        └── reference_files/       # Downloaded reference output files per task
            ├── 0/
            │   ├── 0_ref_0.xlsx
            │   └── ...
            └── ...
    """

    def __init__(self):
        self._tasks: List[FinchTask] = []
        self._samples: List[EvaluationSample] = []

    def load(self, dataset_config=None) -> None:
        """Load Finch tasks from local data directory.

        Args:
            dataset_config: Optional DatasetConfig with dataset_folder setting.
        """
        data_dir = self._resolve_data_dir(dataset_config)
        tasks_path = os.path.join(data_dir, "finch_tasks.json")
        source_dir = os.path.join(data_dir, "source_files")
        reference_dir = os.path.join(data_dir, "reference_files")

        if not os.path.isfile(tasks_path):
            print(
                f"[GLASS] ERROR: Finch task metadata not found at: {tasks_path}\n"
                f"  Please download the dataset first:\n"
                f"    python scripts/download_finch.py\n"
                f"  See: https://huggingface.co/datasets/FinWorkBench/Finch",
                file=sys.stderr,
            )
            raise FileNotFoundError(f"Finch task metadata not found: {tasks_path}")

        with open(tasks_path, encoding="utf-8") as f:
            raw_tasks = json.load(f)

        logger.info("Loading %d Finch tasks from %s", len(raw_tasks), tasks_path)

        for raw in raw_tasks:
            task = self._parse_task(raw, source_dir, reference_dir)
            self._tasks.append(task)

        self._samples = [self._task_to_sample(t) for t in self._tasks]
        logger.info("Loaded %d Finch tasks (%d samples)", len(self._tasks), len(self._samples))

    def get_samples(self) -> List[EvaluationSample]:
        """Return tasks as EvaluationSample for standard GLASS pipeline compatibility.

        The instruction becomes the question, reference text becomes gold_answer,
        and source file info is carried in metadata. This is a lossy conversion —
        use get_tasks() for the full accounting pipeline.
        """
        return self._samples

    def get_tasks(self) -> List[FinchTask]:
        """Return the full FinchTask objects for the accounting pipeline.

        This preserves source file paths, reference outputs, task type taxonomy,
        and business domain needed for chain construction and evaluation.
        """
        return self._tasks

    def get_tasks_by_business_type(self) -> Dict[str, List[FinchTask]]:
        """Group tasks by business_type for chain candidate identification."""
        groups: Dict[str, List[FinchTask]] = {}
        for task in self._tasks:
            key = task.business_type or "Unknown"
            groups.setdefault(key, []).append(task)
        return groups

    def get_tasks_by_task_type(self) -> Dict[str, List[FinchTask]]:
        """Group tasks by individual task_type for workflow analysis."""
        groups: Dict[str, List[FinchTask]] = {}
        for task in self._tasks:
            for tt in task.task_types:
                groups.setdefault(tt, []).append(task)
        return groups

    @staticmethod
    def _resolve_data_dir(dataset_config) -> str:
        """Resolve the Finch data directory from config or default."""
        folder = None
        if dataset_config and hasattr(dataset_config, "dataset_folder"):
            folder = dataset_config.dataset_folder

        if folder:
            return str(Path(folder).resolve())
        return str(Path(_DEFAULT_DATA_DIR).resolve())

    @staticmethod
    def _parse_task(raw: dict, source_dir: str, reference_dir: str) -> FinchTask:
        """Parse a raw task dict into a FinchTask model."""
        task_id = str(raw.get("id", ""))

        # Parse source files
        source_file_names = _parse_list_field(raw.get("source_files"))
        source_file_urls = _parse_list_field(raw.get("source_files_urls"))

        # Resolve local paths for source files
        task_source_dir = os.path.join(source_dir, task_id)
        source_file_paths = []
        for fname in source_file_names:
            local_path = os.path.join(task_source_dir, fname)
            source_file_paths.append(local_path)

        # Parse reference outputs
        ref_output = _parse_reference_outputs(raw.get("reference_outputs"))
        ref_file_urls = _parse_list_field(raw.get("reference_file_urls"))

        # Resolve local paths for reference files
        task_ref_dir = os.path.join(reference_dir, task_id)
        reference_output_paths = []
        for fname in ref_output.files:
            local_path = os.path.join(task_ref_dir, fname)
            reference_output_paths.append(local_path)

        # Parse task types
        task_type_raw = raw.get("task_type", "")
        task_types = _parse_task_types(task_type_raw) if isinstance(task_type_raw, str) else _parse_list_field(task_type_raw)

        return FinchTask(
            task_id=task_id,
            instruction=raw.get("instruction_en", ""),
            source_file_paths=source_file_paths,
            source_file_urls=source_file_urls,
            reference_output_paths=reference_output_paths,
            reference_output_urls=ref_file_urls,
            reference_text=ref_output.text,
            task_types=task_types,
            business_type=raw.get("business_type", "Unknown"),
            task_constraints=raw.get("task_constraints"),
            metadata={
                "raw_task_type": task_type_raw,
                "raw_source_files": raw.get("source_files"),
                "raw_reference_outputs": raw.get("reference_outputs"),
            },
        )

    @staticmethod
    def _task_to_sample(task: FinchTask) -> EvaluationSample:
        """Convert a FinchTask to an EvaluationSample for GLASS compatibility.

        This is a lossy conversion: spreadsheet content is not inlined into
        the prompt (that would require reading .xlsx files). The context_prompt
        contains the instruction + source file listing.
        """
        # Build a lightweight context prompt listing source files
        source_listing = "\n".join(
            f"  - {os.path.basename(p)}" for p in task.source_file_paths
        ) if task.source_file_paths else "  (none)"

        context_prompt = (
            f"Task: {task.instruction}\n\n"
            f"Source files:\n{source_listing}\n\n"
            f"Task types: {', '.join(task.task_types)}\n"
            f"Business domain: {task.business_type}"
        )
        if task.task_constraints:
            context_prompt += f"\nConstraints: {task.task_constraints}"

        # Use reference text as gold answer, or a note about reference files
        gold_answer = task.reference_text or ""
        if not gold_answer and task.reference_output_paths:
            gold_answer = f"[Reference output in files: {', '.join(os.path.basename(p) for p in task.reference_output_paths)}]"

        return EvaluationSample(
            sample_id=task.task_id,
            domain=task.business_type,
            question=task.instruction,
            gold_answer=gold_answer,
            context_prompt=context_prompt,
            input_tokens=len(context_prompt) // 4,
            metadata={
                "task_types": task.task_types,
                "business_type": task.business_type,
                "source_file_paths": task.source_file_paths,
                "reference_output_paths": task.reference_output_paths,
                "task_constraints": task.task_constraints,
            },
        )
