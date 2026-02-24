import pytest

from glass.datasets.base import EvaluationSample
from glass.systems.base import RawOutput
from glass.metrics.bert_score import BertScoreF1Metric


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


def test_bert_score_metric(sample):
    metric = BertScoreF1Metric()
    
    # Perfect match
    out_perfect = create_output("The sky is blue.")
    # Use distilbert-base-uncased for fast unit test
    score_perfect = metric.compute(out_perfect, sample, model_type="distilbert-base-uncased")
    assert score_perfect > 0.9  # BERTScore won't always be exactly 1.0 but very close for exact match
    
    # Semantic match
    out_semantic = create_output("blue is the color of the sky")
    score_semantic = metric.compute(out_semantic, sample, model_type="distilbert-base-uncased")
    assert score_semantic > 0.8
    
    # Error type
    out_err = create_output("The sky is blue.", error_type="timeout")
    score_err = metric.compute(out_err, sample, model_type="distilbert-base-uncased")
    assert score_err is None
