from unittest.mock import MagicMock, patch

import pytest

from glass.config.schema import SystemConfig
from glass.datasets.base import EvaluationSample
from glass.systems.claude import ClaudeSystem
from glass.systems.structured_harness import StructuredHarnessSystem


@pytest.fixture
def sample():
    return EvaluationSample(
        sample_id="1",
        domain="test",
        question="q",
        gold_answer="a",
        context_prompt="prompt",
        input_tokens=10,
        metadata={},
    )


@patch("glass.systems.base.subprocess.Popen")
def test_claude_command(mock_popen, sample):
    process_mock = MagicMock()
    process_mock.communicate.return_value = (b"out", b"err")
    process_mock.returncode = 0
    mock_popen.return_value = process_mock

    config = SystemConfig(name="test", type="claude", command=["claude", "-p"])
    sys = ClaudeSystem(config)
    sys.generate(sample)

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert args[0] == ["claude", "-p"]
    assert kwargs["stdin"] is not None


@patch("glass.systems.base.subprocess.Popen")
def test_structured_harness_command(mock_popen, sample):
    process_mock = MagicMock()
    process_mock.communicate.return_value = (b"out", b"err")
    process_mock.returncode = 0
    mock_popen.return_value = process_mock

    config = SystemConfig(
        name="test",
        type="structured_harness",
        model="sonnet",
        harness_config="single_turn_qa",
    )
    sys = StructuredHarnessSystem(config)
    sys.generate(sample)

    mock_popen.assert_called_once()
    args, _ = mock_popen.call_args
    cmd = args[0]
    assert cmd[0] == "structured-harness"
    assert "single_turn_qa.json" in cmd[1]
    assert cmd[2] == "-"
    assert "--model" in cmd
    assert "sonnet" in cmd
