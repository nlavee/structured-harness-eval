import pytest

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput
from glass.metrics.rouge_score import RougeScoreF1Metric, RougeScoreRecallMetric


@pytest.fixture
def sample():
    return EvaluationSample(
        sample_id="test_001",
        domain="Legal",
        question="What color is the sky?",
        gold_answer="The sky is blue.",
        context_prompt="<doc>Context</doc> What color is the sky?",
        input_tokens=10,
        metadata={}
    )


def create_output(text: str, error_type=None) -> RawOutput:
    return RawOutput(
        sample_id="test_001",
        system_name="test_sys",
        command=["test"],
        prompt="prompt",
        output=text,
        latency_s=1.0,
        exit_code=0 if not error_type else 1,
        stderr="",
        error_type=error_type,
        timestamp="now"
    )


def test_rouge_f1_metric(sample):
    metric = RougeScoreF1Metric()
    
    # Perfect match
    out_perfect = create_output("The sky is blue.")
    score_perfect = metric.compute(out_perfect, sample)
    assert score_perfect == 1.0
    
    # Partial match
    out_partial = create_output("It is blue.")
    score_partial = metric.compute(out_partial, sample)
    assert 0.0 < score_partial < 1.0
    
    # Error type
    out_err = create_output("The sky is blue.", error_type="timeout")
    score_err = metric.compute(out_err, sample)
    assert score_err is None


def test_rouge_recall_metric(sample):
    metric = RougeScoreRecallMetric()
    
    # Perfect match
    out_perfect = create_output("The sky is blue.")
    score_perfect = metric.compute(out_perfect, sample)
    assert score_perfect == 1.0
    
    # Partial match
    out_partial = create_output("It is blue.")
    score_partial = metric.compute(out_partial, sample)
    assert 0.0 < score_partial < 1.0
