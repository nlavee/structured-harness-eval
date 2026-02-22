import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import sys

# Add research_harness to path
sys.path.append(str(Path(__file__).parent.parent / "research_harness"))
import synthesizer

@pytest.fixture
def sample_payload():
    return {
        "metadata": {"paired_sample_n": 10, "systems": ["sys1", "sys2"], "runs": ["run1", "run2"]},
        "global_statistics": {
            "sys1": {"judge_score": {"n": 10, "mean": 0.8, "ci_low": 0.7, "ci_high": 0.9}},
            "sys2": {"judge_score": {"n": 10, "mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}},
        },
        "domain_statistics": {},
        "divergence_pairs_ap_rh4": [
            {"sample_id": "1", "domain": "Legal", "sys1_judge_score": 1.0, "sys2_judge_score": 0.0}
        ],
        "joined_dataframe": []
    }

def test_generate_prompt_context(sample_payload):
    """Test forming context string without exceptions."""
    context_str = synthesizer.generate_prompt_context(sample_payload)
    assert "TOTAL PAIRED SAMPLES ANALYZED: 10" in context_str
    assert "SYSTEMS COMPARED: sys1, sys2" in context_str
    assert "sys1" in context_str
    assert "judge_score: 0.80" in context_str
    assert "Divergence Sample 1" in context_str

@patch('synthesizer.litellm.completion')
def test_synthesizer_main(mock_completion, tmp_path, sample_payload):
    """Test AP-RH4 thought logging and main execution."""
    # Setup mock LLM response
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content="<thought>I am thinking</thought>\nHere is the insight."))
    ]
    mock_completion.return_value = mock_resp
    
    # Write payload file
    data_file = tmp_path / "data.json"
    import json
    data_file.write_text(json.dumps(sample_payload))
    
    out_dir = tmp_path / "insights"
    
    # We patch sys.argv to mock argparse
    test_args = ["synthesizer.py", "--data", str(data_file), "--out-dir", str(out_dir)]
    
    with patch.object(sys, 'argv', test_args):
        synthesizer.main()
        
    # Check outputs were created
    log_files = list((out_dir / "logs").glob("synthesizer_thought_*.txt"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text()
    
    assert "I am thinking" in log_content
    assert "=== PROMPT SENT TO LLM ===" in log_content
    
    insight_files = list(out_dir.glob("insights_*.md"))
    assert len(insight_files) == 1
    insight_content = insight_files[0].read_text()
    
    assert "Here is the insight" in insight_content
    # Ensure thought tags are stripped from output
    assert "<thought>" not in insight_content
