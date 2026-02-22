"""Tests for reasoning_depth metric."""

from glass.metrics.reasoning_depth import ReasoningDepthMetric
from tests.conftest import make_output, make_sample


class TestReasoningDepth:
    def test_returns_none_on_error(self):
        m = ReasoningDepthMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_empty_output(self):
        m = ReasoningDepthMetric()
        assert m.compute(make_output(""), make_sample()) == 0.0

    def test_no_markers(self):
        m = ReasoningDepthMetric()
        assert m.compute(make_output("Paris"), make_sample()) == 0.0

    def test_causal_markers(self):
        m = ReasoningDepthMetric()
        result = m.compute(
            make_output("Paris because it is the capital. Therefore it is correct."),
            make_sample(),
        )
        assert result > 0.0

    def test_enumeration_markers(self):
        m = ReasoningDepthMetric()
        result = m.compute(
            make_output("1. First point.\n2. Second point.\n3. Third point."),
            make_sample(),
        )
        assert result > 0.0

    def test_evidence_markers(self):
        m = ReasoningDepthMetric()
        result = m.compute(
            make_output("According to the document, the answer is Paris. Based on the report, this is confirmed."),
            make_sample(),
        )
        assert result > 0.0

    def test_capped_at_one(self):
        m = ReasoningDepthMetric()
        text = "Because X. Therefore Y. Thus Z. Hence A. Consequently B. " * 5
        text += "According to the document. Based on the evidence. " * 5
        result = m.compute(make_output(text), make_sample())
        assert result == 1.0

    def test_combined_markers(self):
        m = ReasoningDepthMetric()
        result = m.compute(
            make_output(
                "According to the document, the answer is large because it includes many cities. "
                "Therefore, the total is significant."
            ),
            make_sample(),
        )
        assert result > 0.1

    def test_category_is_behavioral(self):
        assert ReasoningDepthMetric().category == "behavioral"
