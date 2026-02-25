"""Extended system tests: Gemini, Codex, Stub; command construction; AP-5/AP-6/AP-8."""

from unittest.mock import MagicMock, patch

import pytest

from glass.config.schema import SystemConfig
from glass.datasets.base import EvaluationSample
from glass.systems.codex import CodexSystem
from glass.systems.gemini import GeminiSystem
from glass.systems.stub import StubSystem


@pytest.fixture
def sample():
    return EvaluationSample(
        sample_id="42",
        domain="Legal",
        question="q",
        gold_answer="gold answer",
        context_prompt="long prompt here",
        input_tokens=100,
        metadata={},
    )


# --------------------------------------------------------------------------- #
# Gemini system                                                               #
# --------------------------------------------------------------------------- #


@patch("glass.systems.base.subprocess.Popen")
def test_gemini_default_command(mock_popen, sample):
    proc = MagicMock()
    proc.communicate.return_value = (b"result", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="gemini-cli", type="gemini")
    sys = GeminiSystem(config)
    sys.generate(sample)

    args, kwargs = mock_popen.call_args
    assert args[0] == ["gemini", "--output-format", "text"]
    assert kwargs.get("shell") is not True  # AP-5: no shell=True
    assert kwargs["stdin"] is not None  # AP-6: prompt via stdin


@patch("glass.systems.base.subprocess.Popen")
def test_gemini_command_in_output(mock_popen, sample):
    """AP-8: RawOutput must contain the exact command used."""
    proc = MagicMock()
    proc.communicate.return_value = (b"answer", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="gemini-cli", type="gemini")
    sys = GeminiSystem(config)
    output = sys.generate(sample)

    assert output.command == ["gemini", "--output-format", "text"]


@patch("glass.systems.base.subprocess.Popen")
def test_gemini_cot_capture(mock_popen, sample):
    """When quiet=False, stderr is captured as chain_of_thought."""
    proc = MagicMock()
    proc.communicate.return_value = (b"answer", b"thinking about it...")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="gemini-cli", type="gemini", quiet=False)
    sys = GeminiSystem(config)
    output = sys.generate(sample)

    assert output.chain_of_thought == "thinking about it..."


@patch("glass.systems.base.subprocess.Popen")
def test_gemini_stream_json_parsing(mock_popen, sample):
    """Test correctly parsing Gemini CLI stream-json output."""
    proc = MagicMock()
    
    # Mocking a sequence of JSONL events
    mock_output = (
        '{"type": "init", "session": "v1"}\n'
        '{"type": "message", "role": "assistant", "content": "I am thinking", "thinking": "Internal thought process"}\n'
        '{"type": "message", "role": "assistant", "content": " The final answer is 42."}\n'
        '{"type": "result", "status": "success"}\n'
    )
    
    proc.communicate.return_value = (mock_output.encode("utf-8"), b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="gemini-cli", type="gemini", output_format="stream-json")
    sys = GeminiSystem(config)
    output = sys.generate(sample)

    assert output.output == "I am thinking The final answer is 42."
    assert output.chain_of_thought == "Internal thought process"


# --------------------------------------------------------------------------- #
# Codex system                                                                #
# --------------------------------------------------------------------------- #


@patch("glass.systems.base.subprocess.Popen")
def test_codex_respects_config_model(mock_popen, sample):
    """Codex should use config.model, not hardcoded gpt-5."""
    proc = MagicMock()
    proc.communicate.return_value = (b"out", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="codex-cli", type="codex", model="gpt-4.1")
    sys = CodexSystem(config)
    output = sys.generate(sample)

    args, _ = mock_popen.call_args
    cmd = args[0]
    assert "gpt-4.1" in cmd
    assert output.command == cmd


@patch("glass.systems.base.subprocess.Popen")
def test_codex_default_model(mock_popen, sample):
    proc = MagicMock()
    proc.communicate.return_value = (b"", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="codex-cli", type="codex")
    sys = CodexSystem(config)
    sys.generate(sample)

    args, _ = mock_popen.call_args
    cmd = args[0]
    assert "gpt-5" in cmd  # fallback default


@patch("glass.systems.base.subprocess.Popen")
def test_codex_stdin_flag(mock_popen, sample):
    """Codex must end command with '-' to read from stdin."""
    proc = MagicMock()
    proc.communicate.return_value = (b"out", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="codex-cli", type="codex")
    sys = CodexSystem(config)
    sys.generate(sample)

    args, _ = mock_popen.call_args
    cmd = args[0]
    assert cmd[-1] == "-"


@patch("glass.systems.base.subprocess.Popen")
def test_codex_quiet_mode(mock_popen, sample):
    """When quiet=True, --quiet flag should be present."""
    proc = MagicMock()
    proc.communicate.return_value = (b"out", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="codex-cli", type="codex", quiet=True)
    sys = CodexSystem(config)
    sys.generate(sample)

    args, _ = mock_popen.call_args
    cmd = args[0]
    assert "--quiet" in cmd
    assert cmd[-1] == "-"  # stdin flag still last


@patch("glass.systems.base.subprocess.Popen")
def test_codex_no_quiet_by_default(mock_popen, sample):
    """Default (quiet=False) should not include --quiet."""
    proc = MagicMock()
    proc.communicate.return_value = (b"out", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="codex-cli", type="codex")
    sys = CodexSystem(config)
    sys.generate(sample)

    args, _ = mock_popen.call_args
    cmd = args[0]
    assert "--quiet" not in cmd


@patch("glass.systems.base.subprocess.Popen")
def test_codex_cot_capture(mock_popen, sample):
    """When quiet=False, stderr is captured as chain_of_thought."""
    proc = MagicMock()
    proc.communicate.return_value = (b"final answer", b"reasoning step 1\nreasoning step 2")
    proc.returncode = 0
    mock_popen.return_value = proc

    config = SystemConfig(name="codex-cli", type="codex", quiet=False)
    sys = CodexSystem(config)
    output = sys.generate(sample)

    assert output.chain_of_thought == "reasoning step 1\nreasoning step 2"
    assert output.output == "final answer"


# --------------------------------------------------------------------------- #
# Stub system                                                                 #
# --------------------------------------------------------------------------- #


def test_stub_returns_gold_answer(sample):
    config = SystemConfig(name="stub", type="stub")
    sys = StubSystem(config)
    output = sys.generate(sample)

    assert output.output == sample.gold_answer
    assert output.exit_code == 0
    assert output.error_type is None


def test_stub_includes_command_field(sample):
    """AP-8: stub must include command in RawOutput."""
    config = SystemConfig(name="stub", type="stub")
    sys = StubSystem(config)
    output = sys.generate(sample)

    assert isinstance(output.command, list)
    assert len(output.command) > 0


# --------------------------------------------------------------------------- #
# Encoding safety (AP-27)                                                     #
# --------------------------------------------------------------------------- #


@patch("glass.systems.base.subprocess.Popen")
def test_non_utf8_stdout_handled(mock_popen, sample):
    """AP-27: Non-UTF-8 bytes in stdout must not crash (errors='replace')."""
    proc = MagicMock()
    # Inject invalid UTF-8 sequence
    proc.communicate.return_value = (b"\xff\xfe invalid bytes", b"")
    proc.returncode = 0
    mock_popen.return_value = proc

    from glass.systems.claude import ClaudeSystem

    config = SystemConfig(name="claude-code", type="claude", quiet=True)
    sys = ClaudeSystem(config)
    output = sys.generate(sample)

    assert output.output  # Should not raise; replacement chars used
    assert output.error_type is None  # Non-UTF8 alone isn't an error


# --------------------------------------------------------------------------- #
# Timeout handling                                                            #
# --------------------------------------------------------------------------- #


@patch("glass.systems.base.subprocess.Popen")
def test_timeout_produces_error_type(mock_popen, sample):
    import subprocess

    proc = MagicMock()
    proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=1)
    proc.kill = MagicMock()
    mock_popen.return_value = proc

    from glass.systems.claude import ClaudeSystem

    config = SystemConfig(name="claude-code", type="claude", timeout_s=1)
    sys = ClaudeSystem(config)
    output = sys.generate(sample)

    assert output.error_type == "timeout"
    assert output.exit_code == 124
    assert output.output == ""
    proc.kill.assert_called_once()
