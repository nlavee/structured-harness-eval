"""Shared fixtures and helpers for metric tests."""

import pytest

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput


def make_output(output_text="answer", error_type=None, exit_code=0, latency=1.0):
    """Create a RawOutput fixture with sensible defaults."""
    return RawOutput(
        sample_id="x",
        system_name="s",
        command=["stub"],
        prompt="",
        output=output_text,
        latency_s=latency,
        exit_code=exit_code,
        stderr="",
        error_type=error_type,
        timestamp="t",
    )


def make_sample(gold="answer", question="What is the answer?"):
    """Create an EvaluationSample fixture with sensible defaults."""
    return EvaluationSample(
        sample_id="x",
        domain="d",
        question=question,
        gold_answer=gold,
        context_prompt="ctx",
        input_tokens=10,
        metadata={},
    )
