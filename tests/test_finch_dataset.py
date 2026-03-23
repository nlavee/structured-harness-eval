"""Tests for the Finch dataset adapter (glass/datasets/finch.py).

Follows the same patterns as test_dataset.py for AA-LCR.
"""

import json
import os
import tempfile

import pytest

from glass.datasets.finch import (
    FinchAdapter,
    FinchReferenceOutput,
    FinchTask,
    _parse_list_field,
    _parse_reference_outputs,
    _parse_task_types,
)
from glass.datasets.registry import get_dataset_class


# ── Registry ─────────────────────────────────────────────────────────────

def test_registry():
    cls = get_dataset_class("finch")
    assert cls == FinchAdapter


# ── FinchTask model ──────────────────────────────────────────────────────

def test_finch_task_minimal():
    task = FinchTask(
        task_id="42",
        instruction="Complete the balance sheet",
        source_file_paths=[],
        source_file_urls=[],
        reference_output_paths=[],
        reference_output_urls=[],
        task_types=["Validation/Review"],
        business_type="Corporate Finance",
    )
    assert task.task_id == "42"
    assert task.business_type == "Corporate Finance"
    assert task.reference_text is None
    assert task.task_constraints is None


def test_finch_task_full():
    task = FinchTask(
        task_id="7",
        instruction="Reconcile the GL",
        source_file_paths=["/data/7/src.xlsx"],
        source_file_urls=["https://example.com/src.xlsx"],
        reference_output_paths=["/data/7/ref.xlsx"],
        reference_output_urls=["https://example.com/ref.xlsx"],
        reference_text="Total assets: $1,234,567",
        task_types=["Calculation", "Validation/Review"],
        business_type="Energy/Utilities",
        task_constraints="Must use GAAP",
        metadata={"custom": "value"},
    )
    assert len(task.task_types) == 2
    assert task.task_constraints == "Must use GAAP"
    assert task.metadata["custom"] == "value"


# ── Parsing helpers ──────────────────────────────────────────────────────

def test_parse_task_types_comma_separated():
    assert _parse_task_types("Validation/Review, Calculation") == [
        "Validation/Review", "Calculation"
    ]


def test_parse_task_types_single():
    assert _parse_task_types("Report & Analysis") == ["Report & Analysis"]


def test_parse_task_types_empty():
    assert _parse_task_types("") == []
    assert _parse_task_types(None) == []


def test_parse_list_field_list():
    assert _parse_list_field(["a.xlsx", "b.csv"]) == ["a.xlsx", "b.csv"]


def test_parse_list_field_json_string():
    assert _parse_list_field('["a.xlsx", "b.csv"]') == ["a.xlsx", "b.csv"]


def test_parse_list_field_semicolon():
    assert _parse_list_field("a.xlsx;b.csv") == ["a.xlsx", "b.csv"]


def test_parse_list_field_none():
    assert _parse_list_field(None) == []


def test_parse_list_field_empty_string():
    assert _parse_list_field("") == []


def test_parse_reference_outputs_dict():
    ref = _parse_reference_outputs({
        "files": ["0_ref_0.xlsx"],
        "text": "Answer is 42",
    })
    assert ref.files == ["0_ref_0.xlsx"]
    assert ref.text == "Answer is 42"


def test_parse_reference_outputs_json_string():
    ref = _parse_reference_outputs('{"files": ["out.xlsx"], "text": null}')
    assert ref.files == ["out.xlsx"]
    assert ref.text is None


def test_parse_reference_outputs_plain_text():
    ref = _parse_reference_outputs("The total is $500")
    assert ref.files == []
    assert ref.text == "The total is $500"


def test_parse_reference_outputs_none():
    ref = _parse_reference_outputs(None)
    assert ref.files == []
    assert ref.text is None


# ── Adapter load ─────────────────────────────────────────────────────────

def _make_finch_data(tmpdir, tasks):
    """Create a mock Finch data directory with given tasks."""
    data_dir = os.path.join(tmpdir, "Finch")
    os.makedirs(data_dir, exist_ok=True)

    # Write tasks metadata
    with open(os.path.join(data_dir, "finch_tasks.json"), "w") as f:
        json.dump(tasks, f)

    # Create source and reference directories
    source_dir = os.path.join(data_dir, "source_files")
    ref_dir = os.path.join(data_dir, "reference_files")

    for task in tasks:
        task_id = str(task.get("id", ""))

        # Create source files
        src_files = task.get("source_files", [])
        if isinstance(src_files, str):
            try:
                src_files = json.loads(src_files)
            except json.JSONDecodeError:
                src_files = src_files.split(";")

        task_src_dir = os.path.join(source_dir, task_id)
        os.makedirs(task_src_dir, exist_ok=True)
        for fname in src_files:
            with open(os.path.join(task_src_dir, fname), "w") as f:
                f.write(f"mock content for {fname}")

        # Create reference files
        ref_outputs = task.get("reference_outputs", {})
        if isinstance(ref_outputs, dict):
            ref_files = ref_outputs.get("files", [])
        else:
            ref_files = []

        task_ref_dir = os.path.join(ref_dir, task_id)
        os.makedirs(task_ref_dir, exist_ok=True)
        for fname in ref_files:
            with open(os.path.join(task_ref_dir, fname), "w") as f:
                f.write(f"mock reference for {fname}")

    return data_dir


def test_adapter_load_basic():
    mock_tasks = [
        {
            "id": "0",
            "instruction_en": "Complete the balance sheet validation",
            "source_files": ["0_src_0.xlsx"],
            "source_files_urls": ["https://example.com/0_src_0.xlsx"],
            "reference_outputs": {
                "files": ["0_ref_0.xlsx"],
                "text": "Total assets equal total liabilities plus equity",
            },
            "reference_file_urls": ["https://example.com/0_ref_0.xlsx"],
            "task_type": "Validation/Review, Calculation",
            "business_type": "Corporate Finance",
            "task_constraints": None,
        },
        {
            "id": "1",
            "instruction_en": "Calculate NPV for the project",
            "source_files": ["1_src_0.xlsx", "1_src_1.csv"],
            "source_files_urls": [],
            "reference_outputs": {
                "files": [],
                "text": "NPV = $1,234,567",
            },
            "reference_file_urls": [],
            "task_type": "Calculation & Financial Modeling",
            "business_type": "Energy/Utilities",
            "task_constraints": "Use 8% discount rate",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = _make_finch_data(tmpdir, mock_tasks)

        class MockConfig:
            dataset_folder = data_dir

        adapter = FinchAdapter()
        adapter.load(dataset_config=MockConfig())

        # get_tasks() returns FinchTask objects
        tasks = adapter.get_tasks()
        assert len(tasks) == 2

        t0 = tasks[0]
        assert t0.task_id == "0"
        assert t0.instruction == "Complete the balance sheet validation"
        assert t0.task_types == ["Validation/Review", "Calculation"]
        assert t0.business_type == "Corporate Finance"
        assert t0.reference_text == "Total assets equal total liabilities plus equity"
        assert len(t0.source_file_paths) == 1
        assert t0.source_file_paths[0].endswith("0_src_0.xlsx")

        t1 = tasks[1]
        assert t1.task_id == "1"
        assert len(t1.source_file_paths) == 2
        assert t1.task_constraints == "Use 8% discount rate"

        # get_samples() returns EvaluationSample objects
        samples = adapter.get_samples()
        assert len(samples) == 2

        s0 = samples[0]
        assert s0.sample_id == "0"
        assert s0.domain == "Corporate Finance"
        assert s0.question == "Complete the balance sheet validation"
        assert "Total assets equal total liabilities plus equity" in s0.gold_answer
        assert "Validation/Review" in s0.context_prompt

        s1 = samples[1]
        assert s1.sample_id == "1"
        assert s1.metadata["task_types"] == ["Calculation & Financial Modeling"]


def test_adapter_get_tasks_by_business_type():
    mock_tasks = [
        {"id": "0", "instruction_en": "Task A", "source_files": [],
         "task_type": "Calculation", "business_type": "Finance"},
        {"id": "1", "instruction_en": "Task B", "source_files": [],
         "task_type": "Validation", "business_type": "Finance"},
        {"id": "2", "instruction_en": "Task C", "source_files": [],
         "task_type": "Report", "business_type": "Energy"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = _make_finch_data(tmpdir, mock_tasks)

        class MockConfig:
            dataset_folder = data_dir

        adapter = FinchAdapter()
        adapter.load(dataset_config=MockConfig())

        groups = adapter.get_tasks_by_business_type()
        assert len(groups) == 2
        assert len(groups["Finance"]) == 2
        assert len(groups["Energy"]) == 1


def test_adapter_get_tasks_by_task_type():
    mock_tasks = [
        {"id": "0", "instruction_en": "A", "source_files": [],
         "task_type": "Calculation, Validation", "business_type": "X"},
        {"id": "1", "instruction_en": "B", "source_files": [],
         "task_type": "Validation", "business_type": "X"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = _make_finch_data(tmpdir, mock_tasks)

        class MockConfig:
            dataset_folder = data_dir

        adapter = FinchAdapter()
        adapter.load(dataset_config=MockConfig())

        groups = adapter.get_tasks_by_task_type()
        assert len(groups["Validation"]) == 2
        assert len(groups["Calculation"]) == 1


def test_adapter_missing_tasks_file():
    class MockConfig:
        dataset_folder = "/nonexistent/finch/path"

    adapter = FinchAdapter()
    with pytest.raises(FileNotFoundError, match="Finch task metadata not found"):
        adapter.load(dataset_config=MockConfig())


def test_adapter_sample_gold_answer_fallback():
    """When reference_text is empty, gold_answer should mention reference files."""
    mock_tasks = [
        {
            "id": "5",
            "instruction_en": "Format the spreadsheet",
            "source_files": ["5_src_0.xlsx"],
            "reference_outputs": {"files": ["5_ref_0.xlsx"], "text": None},
            "task_type": "Structuring/Formatting",
            "business_type": "Retail",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = _make_finch_data(tmpdir, mock_tasks)

        class MockConfig:
            dataset_folder = data_dir

        adapter = FinchAdapter()
        adapter.load(dataset_config=MockConfig())

        samples = adapter.get_samples()
        assert "5_ref_0.xlsx" in samples[0].gold_answer


def test_adapter_empty_dataset():
    """Adapter should handle an empty task list gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = _make_finch_data(tmpdir, [])

        class MockConfig:
            dataset_folder = data_dir

        adapter = FinchAdapter()
        adapter.load(dataset_config=MockConfig())

        assert adapter.get_tasks() == []
        assert adapter.get_samples() == []
