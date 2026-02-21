from unittest.mock import patch

import pytest

from glass.datasets.base import EvaluationSample
from glass.judges.llm import LLMJudge
from glass.metrics.exact_match import ExactMatchMetric
from glass.metrics.refusal import RefusalMetric
from glass.metrics.soft_recall import SoftRecallMetric
from glass.metrics.verbosity import VerbosityMetric
from glass.systems.base import RawOutput


@pytest.fixture
def data():
    sample = EvaluationSample(
        sample_id="1",
        domain="d",
        question="q",
        gold_answer="Paris",
        context_prompt="c",
        input_tokens=10,
        metadata={},
    )
    output = RawOutput(
        sample_id="1",
        system_name="s",
        command=["stub"],
        prompt="",
        output="paris",
        latency_s=1,
        exit_code=0,
        stderr="",
        timestamp="t",
    )
    return sample, output


def test_exact_match(data):
    sample, output = data
    m = ExactMatchMetric()
    assert m.compute(output, sample) == 1.0

    output.output = "London"
    assert m.compute(output, sample) == 0.0


def test_soft_recall(data):
    sample, output = data
    m = SoftRecallMetric()
    output.output = "Paris is nice"
    # gold="Paris". tokens=["paris"]. pred=["paris", "is", "nice"].
    # overlap=1. len(gold)=1. score=1.0
    assert m.compute(output, sample) == 1.0

    sample.gold_answer = "Paris France"
    # gold=["paris", "france"]. pred=["paris", "is", "nice"].
    # overlap=1. len=2. score=0.5
    assert m.compute(output, sample) == 0.5


def test_verbosity(data):
    sample, output = data
    m = VerbosityMetric()
    sample.gold_answer = "1234"
    output.output = "12"
    assert m.compute(output, sample) == 0.5


def test_refusal(data):
    sample, output = data
    m = RefusalMetric()
    output.output = "Sure, here is the answer."
    assert m.compute(output, sample) == 0.0

    output.output = "I cannot answer that."
    assert m.compute(output, sample) == 1.0


def test_judge_mock():
    with patch("glass.judges.llm.LLMJudge._call_llm") as mock_call:
        with patch("glass.judges.llm.nltk.download"):  # Mock nltk download to avoid net
            j = LLMJudge("openai", "gpt-4")

            mock_call.return_value = "CORRECT"
            score, _ = j.evaluate_correctness("q", "a", "p")
            assert score == 1.0

            mock_call.return_value = "INCORRECT"
            score, _ = j.evaluate_correctness("q", "a", "p")
            assert score == 0.0
