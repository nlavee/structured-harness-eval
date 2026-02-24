import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import os

import pytest

from glass.config.schema import Config, ExperimentConfig, DatasetConfig, OutputConfig
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
    mock_metric.is_batchable = False
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

    # Verify Eval (Metrics now receive the judge)
    mock_metric.compute.assert_called_once()
    _, kwargs = mock_metric.compute.call_args
    assert kwargs["judge"] == mock_judge
    assert "judge_outputs" in kwargs


@pytest.fixture
def mock_config_helpers(tmp_path):
    return Config(
        experiment=ExperimentConfig(name="test", run_id=None, seed=42),
        dataset=DatasetConfig(name="aa_lcr", samples=1, domains=None, dataset_folder="."),
        systems=[],
        metrics=[],
        judges={"strategy": "rotation", "fixed": {"provider": "openai", "model": "gpt"}, 
                "rotation": {"anthropic": {"provider": "google", "model": "test"}, "google": {"provider": "anthropic", "model": "test"}, "openai": {"provider": "anthropic", "model": "test"}}, 
                "templates": {"correctness": "a", "hallucination": "b"}},
        statistics={"bootstrap_resamples": 1, "alpha": 0.05, "primary_test": "t", "secondary_test": "t"},
        output=OutputConfig(runs_dir=str(tmp_path), log_level="INFO")
    )


@patch("glass.pipeline.RunStore")
@patch("glass.pipeline.CheckpointManager")
@patch("glass.pipeline.Pipeline._setup_run_directory")
def test_determine_run_id_generate_new(mock_setup, mock_checkpoint, mock_store, mock_config_helpers):
    p = Pipeline(mock_config_helpers)
    assert "test" in p.run_id
    assert len(p.run_id) > 10


@patch("glass.pipeline.RunStore")
@patch("glass.pipeline.CheckpointManager")
@patch("glass.pipeline.Pipeline._setup_run_directory")
def test_determine_run_id_branched(mock_setup, mock_checkpoint, mock_store, mock_config_helpers, tmp_path):
    (tmp_path / "base_run").mkdir()
    (tmp_path / "base_run_1").mkdir()
    (tmp_path / "base_run_2").mkdir()

    p = Pipeline(mock_config_helpers, re_evaluate_source="base_run")
    assert p.run_id == "base_run_3"

    p2 = Pipeline(mock_config_helpers, re_evaluate_source="base_run_2")
    assert p2.run_id == "base_run_3"


@patch("glass.pipeline.CheckpointManager")
def test_inherit_from_source(mock_checkpoint, mock_config_helpers, tmp_path):
    runs_dir = tmp_path
    
    # Setup source directory
    source_dir = runs_dir / "my_source"
    source_dir.mkdir()
    
    (source_dir / "inference").mkdir()
    (source_dir / "inference" / "tester.json").write_text("{}")
    (source_dir / "manifest.json").write_text("{}")
    (source_dir / "glass.log").write_text("logs")
    
    # Initialize pipeline with branch request
    p = Pipeline(mock_config_helpers, re_evaluate_source="my_source")
    
    # Verify copy occurred
    target_dir = runs_dir / "my_source_1"
    assert target_dir.exists()
    assert (target_dir / "inference" / "tester.json").exists()
    assert (target_dir / "manifest.json").exists()
    assert (target_dir / "glass.log").exists()
    assert (target_dir / "config.yaml").exists()


@patch("glass.pipeline.RunStore")
@patch("glass.pipeline.CheckpointManager")
@patch("glass.pipeline.Pipeline._setup_run_directory")
def test_load_data_and_systems(mock_setup, mock_checkpoint, mock_store, mock_config_helpers):
    # Mock systems to avoid instantiation errors
    mock_config_helpers.systems = []
    
    with patch("glass.pipeline.get_dataset_class") as mock_ds_cls:
        mock_ds = MagicMock()
        mock_ds.get_samples.return_value = [
            MagicMock(sample_id="1", domain="x"),
            MagicMock(sample_id="2", domain="y")
        ]
        mock_ds_cls.return_value = MagicMock(return_value=mock_ds)
        
        p = Pipeline(mock_config_helpers)
        samples, systems = p._load_data_and_systems()
        assert len(samples) == 1  # From config.dataset.samples = 1
        assert len(systems) == 0
