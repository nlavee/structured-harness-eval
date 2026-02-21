import json
from unittest.mock import MagicMock, patch

import pytest

from glass.config.schema import Config
from glass.pipeline import Pipeline


@pytest.fixture
def mock_config(tmp_path):
    config = {
        "experiment": {"name": "test", "seed": 42},
        "dataset": {"name": "aa_lcr", "samples": 1},
        "systems": [{"name": "s1", "type": "claude"}],
        "metrics": ["exact_match"],
        "judges": {
            "strategy": "fixed",
            "fixed": {"provider": "openai", "model": "gpt-4"},
            "templates": {"correctness": "T", "hallucination": "T"},
        },
        "statistics": {"primary_test": "w", "secondary_test": "t"},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    return Config(**config)


@patch("glass.pipeline.get_dataset_class")
@patch("glass.pipeline.get_system_class")
@patch("glass.pipeline.get_metric_class")
@patch("glass.pipeline.RotationStrategy")
def test_pipeline_flow(mock_strat, mock_metric_cls, mock_sys_cls, mock_ds_cls, mock_config):
    # Mock Dataset
    mock_ds = MagicMock()
    mock_ds.get_samples.return_value = [
        MagicMock(sample_id="1", domain="d", question="q", gold_answer="a", context_prompt="c")
    ]
    mock_ds_cls.return_value = MagicMock(return_value=mock_ds)

    # Mock System Output
    output_data = {
        "sample_id": "1",
        "system_name": "s1",
        "command": ["claude", "--print"],
        "prompt": "c",
        "output": "o",
        "latency_s": 0.1,
        "exit_code": 0,
        "stderr": "",
        "error_type": None,
        "timestamp": "t",
    }
    mock_output = MagicMock(**output_data)
    mock_output.model_dump_json.return_value = json.dumps(output_data)

    # Mock System
    mock_sys = MagicMock()
    mock_sys.config.name = "s1"
    mock_sys.generate.return_value = mock_output
    mock_sys_cls.return_value = MagicMock(return_value=mock_sys)

    # Mock Metric
    mock_metric = MagicMock()
    mock_metric.compute.return_value = 1.0
    mock_metric_cls.return_value = MagicMock(return_value=mock_metric)

    # Mock Judge
    mock_judge = MagicMock()
    mock_judge.evaluate_correctness.return_value = (1.0, "ok")
    mock_judge.evaluate_hallucination.return_value = (0.0, "ok")
    mock_strat.return_value.assign_judge.return_value = mock_judge

    pipeline = Pipeline(mock_config)
    pipeline.run()

    # Verify Inference
    mock_sys.generate.assert_called_once()

    # Verify Eval
    mock_judge.evaluate_correctness.assert_called_once()
