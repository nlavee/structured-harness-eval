import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add research_harness to path
sys.path.append(str(Path(__file__).parent.parent / "research_harness"))
import vision_interpreter

@patch('vision_interpreter.litellm.completion')
@patch('vision_interpreter.encode_image')
def test_vision_interpreter_main(mock_encode, mock_completion, tmp_path):
    """Test AP-RH6 missing visual interpretation logs and markdown rendering."""
    # Setup mock image encoding
    mock_encode.return_value = "base64_encoded_string"
    
    # Setup mock LLM response
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content="<thought>Visualizing data trends</thought>\nThe image shows a clear trend."))
    ]
    mock_completion.return_value = mock_resp
    
    # Mocking filesystem inputs
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "forest_ci_exact_match.png").write_bytes(b"fake image data")
    
    out_dir = tmp_path / "insights"
    
    test_args = ["vision_interpreter.py", "--figures-dir", str(figures_dir), "--out-dir", str(out_dir)]
    
    with patch.object(sys, 'argv', test_args):
        vision_interpreter.main()
        
    # Check files created
    log_files = list((out_dir / "logs").glob("vision_thought_*.txt"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text()
    
    assert "Visualizing data trends" in log_content
    # Now it groups by metric
    assert "=== THOUGHT PROCESS FOR GROUP: exact_match ===" in log_content
    
    insight_files = list(out_dir.glob("visualization_interpretations_*.md"))
    assert len(insight_files) == 1
    insight_content = insight_files[0].read_text()
    
    assert "The image shows a clear trend." in insight_content
    # Check that thought block is removed from the printed output
    assert "<thought>" not in insight_content
    assert "![forest_ci_exact_match.png](figures/forest_ci_exact_match.png)" in insight_content
