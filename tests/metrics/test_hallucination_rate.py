"""Tests for hallucination_rate metric."""

from glass.metrics.hallucination_rate import HallucinationRateMetric
from tests.conftest import make_output, make_sample


class TestHallucinationRate:
    def test_requires_judge_property(self):
        assert HallucinationRateMetric().requires_judge is True

    def test_category_is_correctness(self):
        assert HallucinationRateMetric().category == "correctness"

    def test_returns_none_on_error(self):
        m = HallucinationRateMetric()
        assert m.compute(make_output(error_type="timeout"), make_sample()) is None
