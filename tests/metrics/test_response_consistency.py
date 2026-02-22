"""Tests for response_consistency metric."""

from glass.metrics.response_consistency import ResponseConsistencyMetric
from tests.conftest import make_output, make_sample


class TestResponseConsistency:
    def test_returns_none_on_error(self):
        m = ResponseConsistencyMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_returns_none_on_empty(self):
        m = ResponseConsistencyMetric()
        assert m.compute(make_output(""), make_sample()) is None

    def test_no_entities_vacuously_consistent(self):
        m = ResponseConsistencyMetric()
        assert m.compute(make_output("the answer is yes"), make_sample()) == 1.0

    def test_consistent_entities(self):
        m = ResponseConsistencyMetric()
        result = m.compute(
            make_output("Paris is the capital of France. France includes Paris in its territory."),
            make_sample(),
        )
        assert result > 0.0

    def test_inconsistent_entities(self):
        m = ResponseConsistencyMetric()
        result = m.compute(
            make_output("Paris France 42. London Germany 99."),
            make_sample(),
        )
        assert result < 1.0

    def test_perfect_consistency(self):
        m = ResponseConsistencyMetric()
        result = m.compute(make_output("Paris 42 Paris 42"), make_sample())
        assert result == 1.0

    def test_numbers_detected(self):
        m = ResponseConsistencyMetric()
        result = m.compute(
            make_output("The value is 100 and 200. The result of 100 plus 200 is 300."),
            make_sample(),
        )
        assert result > 0.0

    def test_category_is_behavioral(self):
        assert ResponseConsistencyMetric().category == "behavioral"
