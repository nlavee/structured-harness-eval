"""Tests for confidence_score metric."""

import pytest

from glass.metrics.confidence_score import ConfidenceScoreMetric
from tests.conftest import make_output, make_sample


class TestConfidenceScore:
    def test_returns_none_on_error(self):
        m = ConfidenceScoreMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_neutral_when_no_markers(self):
        m = ConfidenceScoreMetric()
        assert m.compute(make_output("42"), make_sample()) == 0.5

    def test_confident_markers_detected(self):
        m = ConfidenceScoreMetric()
        result = m.compute(make_output("The answer is X. Clearly this demonstrates Y."), make_sample())
        assert result > 0.5

    def test_hedging_markers_detected(self):
        m = ConfidenceScoreMetric()
        result = m.compute(make_output("I think it might be X. Perhaps Y could be true."), make_sample())
        assert result < 0.5

    def test_mixed_markers_balanced(self):
        m = ConfidenceScoreMetric()
        # 1 confident ("clearly") + 1 hedging ("might be") → 0.5
        result = m.compute(make_output("Clearly it might be Paris."), make_sample())
        assert result == pytest.approx(0.5)

    def test_empty_output(self):
        m = ConfidenceScoreMetric()
        assert m.compute(make_output(""), make_sample()) == 0.5

    def test_category_is_behavioral(self):
        assert ConfidenceScoreMetric().category == "behavioral"
