import json
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add research_harness to path
sys.path.append(str(Path(__file__).parent.parent / "research_harness"))
import compare_runs

@pytest.fixture
def mock_runs_dir(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    
    # Run 1: 3 samples, system "sys1"
    run1 = runs_dir / "run1"
    run1.mkdir()
    
    stats1 = {
        "run_config": {"systems": ["sys1"]},
        "system_stats": {"sys1": {"judge_score": {"n": 3, "mean": 0.5}}},
        "per_domain": {"DomainA": {"sys1": {"judge_score": {"n": 2, "mean": 0.6}}}}
    }
    (run1 / "statistics.json").write_text(json.dumps(stats1))
    (run1 / "config.yaml").write_text("config1")
    
    df1 = pd.DataFrame([
        {"sample_id": "1", "domain": "DomainA", "system_name": "sys1", "judge_score": 1.0},
        {"sample_id": "2", "domain": "DomainA", "system_name": "sys1", "judge_score": 0.0},
        {"sample_id": "3", "domain": "DomainB", "system_name": "sys1", "judge_score": 1.0}, # Only in run1
    ])
    df1.to_csv(run1 / "results.csv", index=False)
    
    # Run 2: 2 samples, system "sys2" (Sample 3 is missing -> tests AP-RH1 inner join)
    run2 = runs_dir / "run2"
    run2.mkdir()
    
    stats2 = {
        "run_config": {"systems": ["sys2"]},
        "system_stats": {"sys2": {"judge_score": {"n": 2, "mean": 0.8}}},
        "per_domain": {"DomainA": {"sys2": {"judge_score": {"n": 2, "mean": 0.9}}}}
    }
    (run2 / "statistics.json").write_text(json.dumps(stats2))
    (run2 / "config.yaml").write_text("config2")
    
    df2 = pd.DataFrame([
        {"sample_id": "1", "domain": "DomainA", "system_name": "sys2", "judge_score": 0.0}, # Diverges from sys1
        {"sample_id": "2", "domain": "DomainA", "system_name": "sys2", "judge_score": 0.0},
        # No sample 3
    ])
    df2.to_csv(run2 / "results.csv", index=False)
    
    return runs_dir

def test_enforce_ap_rh1(mock_runs_dir):
    """Test that it performs a strict inner join on sample_id."""
    runs_data = compare_runs.load_runs(["run1", "run2"], base_dir=str(mock_runs_dir))
    joined_df = compare_runs.enforce_ap_rh1(runs_data)
    
    # Should only have samples 1 and 2, sample 3 dropped.
    assert len(joined_df) == 2
    assert "1" in joined_df["sample_id"].astype(str).values
    assert "2" in joined_df["sample_id"].astype(str).values
    assert "3" not in joined_df["sample_id"].astype(str).values
    
    assert "run1_judge_score" in joined_df.columns
    assert "run2_judge_score" in joined_df.columns

def test_extract_global_stats(mock_runs_dir):
    """Test AP-RH5 extracting canonical stats naturally without computing them again."""
    runs_data = compare_runs.load_runs(["run1", "run2"], base_dir=str(mock_runs_dir))
    global_stats = compare_runs.extract_global_stats(runs_data)
    
    assert global_stats["run1"]["judge_score"]["mean"] == 0.5
    assert global_stats["run2"]["judge_score"]["mean"] == 0.8

def test_build_comparison_payload(mock_runs_dir):
    runs_data = compare_runs.load_runs(["run1", "run2"], base_dir=str(mock_runs_dir))
    payload = compare_runs.build_comparison_payload(runs_data)
    
    assert payload["metadata"]["paired_sample_n"] == 2
    assert payload["metadata"]["systems"] == ["run1", "run2"]
    
    # Divergence calculation tested (Sample 1 diverges: run1 got 1.0, run2 got 0.0)
    divergences = payload["divergence_pairs_ap_rh4"]
    assert len(divergences) == 1
    assert divergences[0]["sample_id"] == 1
    
    # Test valid Schema output (from dict conversion)
    assert isinstance(payload, dict)
