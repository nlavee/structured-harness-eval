"""Tests for verbosity metric."""

from glass.metrics.verbosity import VerbosityMetric
from tests.conftest import make_output, make_sample


class TestVerbosity:
    def test_equal_length(self):
        m = VerbosityMetric()
        assert m.compute(make_output("abc"), make_sample(gold="xyz")) == 1.0

    def test_shorter_prediction(self):
        m = VerbosityMetric()
        assert m.compute(make_output("12"), make_sample(gold="1234")) == 0.5

    def test_longer_prediction(self):
        m = VerbosityMetric()
        result = m.compute(make_output("12345"), make_sample(gold="123"))
        assert abs(result - 5 / 3) < 1e-6

    def test_empty_gold_is_none(self):
        m = VerbosityMetric()
        assert m.compute(make_output("some answer"), make_sample(gold="")) is None

    def test_returns_none_on_error(self):
        m = VerbosityMetric()
        assert m.compute(make_output(error_type="api_error"), make_sample()) is None

    def test_category_is_behavioral(self):
        assert VerbosityMetric().category == "behavioral"
