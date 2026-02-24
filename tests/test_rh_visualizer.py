import json
import pytest
from pathlib import Path
import sys
import pandas as pd

# Add research_harness to path
sys.path.append(str(Path(__file__).parent.parent / "research_harness"))
import visualizer
import naming
from schema import AggregatedData

@pytest.fixture
def mock_payload():
    return {
        "metadata": {
            "runs": ["run1", "run2"],
            "systems": ["run1", "run2"],
            "paired_sample_n": 10
        },
        "global_statistics": {
            "run1": {
                "judge_score": {"n": 10, "mean": 0.8, "ci_low": 0.7, "ci_high": 0.9},
                "hallucination_rate": {"n": 10, "mean": 0.1, "ci_low": 0.0, "ci_high": 0.2},
                "soft_recall": {"n": 10, "mean": 0.6},
                "latency_s": {"n": 10, "mean": 5.0},
                "verbosity": {"n": 10, "mean": 1.2}
            },
            "run2": {
                "judge_score": {"n": 10, "mean": 0.6, "ci_low": 0.4, "ci_high": 0.8},
                "hallucination_rate": {"n": 10, "mean": 0.3, "ci_low": 0.1, "ci_high": 0.5},
                "soft_recall": {"n": 10, "mean": 0.4},
                "latency_s": {"n": 10, "mean": 3.0},
                "verbosity": {"n": 10, "mean": 0.8}
            }
        },
        "domain_statistics": {
            "run1": {
                "DomainA": {"judge_score": {"n": 5, "mean": 0.9}}
            },
            "run2": {
                "DomainA": {"judge_score": {"n": 5, "mean": 0.5}}
            }
        },
        "divergence_pairs_ap_rh4": [],
        "joined_dataframe": [
            {"sample_id": "1", "domain": "DomainA", "run1_judge_score": 1.0, "run2_judge_score": 0.0, "run1_hallucination_rate": 0.0, "run2_hallucination_rate": 0.5, "run1_soft_recall": 1.0, "run2_soft_recall": 0.5, "run1_latency_s": 5.0, "run2_latency_s": 3.0},
            {"sample_id": "2", "domain": "DomainA", "run1_judge_score": 1.0, "run2_judge_score": 1.0, "run1_hallucination_rate": 0.1, "run2_hallucination_rate": 0.1, "run1_soft_recall": 0.8, "run2_soft_recall": 0.8, "run1_latency_s": 4.0, "run2_latency_s": 4.0},
        ]
    }

def test_visualizer_pipeline(mock_payload, tmp_path):
    """Test that all visualizer plots render without crashing using mock data."""
    out_dir = tmp_path / "figures"
    out_dir.mkdir()
    
    sys_names = mock_payload["metadata"]["systems"]
    sample_size = mock_payload["metadata"]["paired_sample_n"]
    global_stats = mock_payload["global_statistics"]
    domain_stats = mock_payload["domain_statistics"]
    df = pd.DataFrame(mock_payload["joined_dataframe"])

    # Attempt to plot each function
    forest_path = out_dir / naming.get_plot_filename(naming.PlotType.FOREST, "judge_score")
    visualizer.plot_forest_cis(global_stats, sys_names, "judge_score", forest_path, sample_size)
    assert forest_path.exists()

    violin_path = out_dir / naming.get_plot_filename(naming.PlotType.VIOLIN, "hallucination_rate")
    visualizer.plot_paired_violin(df, sys_names, ["hallucination_rate"], violin_path)
    assert violin_path.exists()

    diff_path = out_dir / naming.get_plot_filename(naming.PlotType.PAIRED_DIFF, "judge_score")
    visualizer.plot_paired_difference(df, sys_names[:2], ["judge_score"], diff_path)
    assert diff_path.exists()

    radar_path = out_dir / naming.get_plot_filename(naming.PlotType.RADAR)
    visualizer.plot_behavior_radar(global_stats, sys_names, radar_path, sample_size)
    assert radar_path.exists()

    heatmap_path = out_dir / naming.get_plot_filename(naming.PlotType.DOMAIN_HEATMAP, "judge_score")
    visualizer.plot_domain_heatmap(domain_stats, sys_names, "judge_score", heatmap_path)
    assert heatmap_path.exists()

def test_visualizer_main(mock_payload, tmp_path):
    """Test the main entry point to ensure it uses naming.py correctly."""
    data_file = tmp_path / "aggregated_data.json"
    data_file.write_text(json.dumps(mock_payload))
    
    out_dir = tmp_path / "figures"
    
    # Mock sys.argv
    test_args = ["visualizer.py", "--data", str(data_file), "--out-dir", str(out_dir)]
    
    from unittest.mock import patch
    with patch.object(sys, 'argv', test_args):
        visualizer.main()
        
    # Check for expected files using naming utility
    # judge_score is in intersection_metrics (keys of win_rate_matrix if we added it, 
    # but in our mock we didn't add win_rate_matrix. Wait.)
    
    # Let's update mock_payload to include some metrics in win_rate_matrix so main() processes them
    mock_payload["win_rate_matrix"] = {"judge_score": {}}
    data_file.write_text(json.dumps(mock_payload))
    
    with patch.object(sys, 'argv', test_args):
        visualizer.main()
        
    expected_forest = naming.get_plot_filename(naming.PlotType.FOREST, "judge_score")
    assert (out_dir / expected_forest).exists()
    
    expected_radar = naming.get_plot_filename(naming.PlotType.RADAR)
    assert (out_dir / expected_radar).exists()
