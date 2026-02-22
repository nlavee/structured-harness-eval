"""Tests for soft_recall metric — AP-3, AP-15 compliance."""

from glass.metrics.soft_recall import SoftRecallMetric
from tests.conftest import make_output, make_sample


class TestSoftRecall:
    def test_perfect_match(self):
        m = SoftRecallMetric()
        assert m.compute(make_output("Paris"), make_sample(gold="Paris")) == 1.0

    def test_superset_prediction_full_recall(self):
        m = SoftRecallMetric()
        # gold=["paris"], pred=["paris", "is", "nice"] → recall=1.0
        assert m.compute(make_output("Paris is nice"), make_sample(gold="Paris")) == 1.0

    def test_partial_match(self):
        m = SoftRecallMetric()
        # gold=["paris", "france"], pred=["paris", "is", "great"] → recall=0.5
        assert m.compute(make_output("Paris is great"), make_sample(gold="Paris France")) == 0.5

    def test_no_match(self):
        m = SoftRecallMetric()
        assert m.compute(make_output("London Berlin"), make_sample(gold="Paris France")) == 0.0

    def test_empty_gold_is_none(self):
        m = SoftRecallMetric()
        assert m.compute(make_output("some answer"), make_sample(gold="")) is None

    def test_empty_prediction_is_zero(self):
        m = SoftRecallMetric()
        assert m.compute(make_output(""), make_sample(gold="Paris")) == 0.0

    def test_returns_none_on_error(self):
        m = SoftRecallMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_category_is_correctness(self):
        assert SoftRecallMetric().category == "correctness"
