"""Unit tests for Finch exploration logic in scripts/explore_finch.py."""

import pytest
from scripts.explore_finch import (
    _parse_task,
    _extract_file_stem_pattern,
    _find_source_subgroups,
    _score_chain_candidate,
)

def test_parse_task_basic():
    raw = {
        "id": "123",
        "instruction_en": "Complete the ledger",
        "task_type": "Data Entry, Calculation",
        "business_type": "Energy",
        "source_files": "file1.xlsx;file2.csv",
    }
    parsed = _parse_task(raw)
    assert parsed["id"] == "123"
    assert parsed["task_types"] == ["Data Entry", "Calculation"]
    assert parsed["business_type"] == "Energy"
    assert parsed["source_files"] == ["file1.xlsx", "file2.csv"]
    assert parsed["source_file_stem"] == "file" # common prefix of file1 and file2

def test_parse_task_json_files():
    raw = {
        "id": "456",
        "source_files": '["a_src_0.xlsx", "a_src_1.xlsx"]',
    }
    parsed = _parse_task(raw)
    assert parsed["source_files"] == ["a_src_0.xlsx", "a_src_1.xlsx"]
    assert parsed["source_file_stem"] == "a_src"

def test_extract_file_stem_pattern():
    assert _extract_file_stem_pattern(["42_src_0.xlsx", "42_src_1.xlsx"]) == "42_src"
    assert _extract_file_stem_pattern(["enron_sept.pdf"]) == "enron_sept"
    assert _extract_file_stem_pattern([]) == ""
    # Multiple files with different names - returns first stem if no prefix
    assert _extract_file_stem_pattern(["abc.xlsx", "xyz.csv"]) == "abc"

def test_find_source_subgroups():
    group = [
        {"id": "1", "source_file_stem": "alpha", "source_files": ["alpha_0.xlsx"]},
        {"id": "2", "source_file_stem": "alpha", "source_files": ["alpha_1.xlsx"]},
        {"id": "3", "source_file_stem": "beta", "source_files": ["beta_0.xlsx"]},
        {"id": "4", "source_file_stem": "", "source_files": ["shared.csv"]},
        {"id": "5", "source_file_stem": "", "source_files": ["shared.csv"]},
    ]
    subgroups = _find_source_subgroups(group)
    # Expected: 
    # Group 1: {1, 2} (shared stem "alpha")
    # Group 2: {3} (stem "beta")
    # Group 3: {4, 5} (shared exact file "shared.csv")
    assert len(subgroups) == 3
    
    # Sort by task_ids to identify them
    subgroups.sort(key=lambda x: x["task_ids"][0])
    
    assert set(subgroups[0]["task_ids"]) == {"1", "2"}
    assert subgroups[0]["shared_pattern"] == "alpha"
    
    assert set(subgroups[1]["task_ids"]) == {"3"}
    assert subgroups[1]["shared_pattern"] == "beta"
    
    assert set(subgroups[2]["task_ids"]) == {"4", "5"}
    assert subgroups[2]["shared_pattern"] == "(no shared pattern)"

def test_score_chain_candidate_full():
    sg = {
        "task_ids": ["1", "2", "3", "4"],
        "shared_pattern": "enron_2001",
        "tasks": [
            {"task_types": ["Data Entry"]},
            {"task_types": ["Calculation"]},
            {"task_types": ["Validation"]},
            {"task_types": ["Report"]},
        ]
    }
    score_info = _score_chain_candidate(sg)
    # Tasks: 4 -> task_count_score = 4/3 = 1.33
    # Macro categories: data_entry, calculation, validation, reporting -> 4 * 1.5 = 6.0
    # Diversity: 4 types -> 4/3 = 1.33
    # Pattern: "enron_2001" -> 1.0
    # Total: ~9.67
    assert score_info["chain_score"] > 9.0
    assert set(score_info["macro_categories_covered"]) == {"data_entry", "calculation", "validation", "reporting"}

def test_score_chain_candidate_minimal():
    sg = {
        "task_ids": ["1"],
        "shared_pattern": "(no shared pattern)",
        "tasks": [
            {"task_types": ["Data Entry"]},
        ]
    }
    score_info = _score_chain_candidate(sg)
    # Tasks: 1 -> 1/3 = 0.33
    # Macro: data_entry -> 1.5
    # Diversity: 1 -> 0.33
    # Pattern: 0.0
    # Total: ~2.16
    assert 2.0 < score_info["chain_score"] < 2.5
    assert score_info["macro_categories_covered"] == ["data_entry"]
