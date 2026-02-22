"""Tests for refusal_rate metric — AP-16 compliance."""

from glass.metrics.refusal import RefusalMetric
from tests.conftest import make_output, make_sample


class TestRefusal:
    def test_no_refusal(self):
        m = RefusalMetric()
        assert m.compute(make_output("The answer is 42."), make_sample()) == 0.0

    def test_refusal_pattern_i_cannot_answer(self):
        m = RefusalMetric()
        assert m.compute(make_output("I cannot answer this question."), make_sample()) == 1.0

    def test_refusal_pattern_case_insensitive(self):
        m = RefusalMetric()
        assert m.compute(make_output("i cannot provide that information"), make_sample()) == 1.0

    def test_explicit_error_type_refusal(self):
        m = RefusalMetric()
        assert m.compute(make_output(error_type="refusal"), make_sample()) == 1.0

    def test_none_on_non_refusal_error(self):
        """Refusal should NOT check patterns on crash/timeout outputs (AP-16)."""
        m = RefusalMetric()
        assert m.compute(make_output(output_text="I'm sorry", error_type="crash"), make_sample()) is None

    def test_none_on_timeout(self):
        m = RefusalMetric()
        assert m.compute(make_output(error_type="timeout"), make_sample()) is None

    def test_i_apologize(self):
        m = RefusalMetric()
        assert m.compute(make_output("I apologize, but I can't help."), make_sample()) == 1.0

    def test_as_ai_language_model(self):
        m = RefusalMetric()
        assert m.compute(make_output("As an AI language model, I cannot do that."), make_sample()) == 1.0
