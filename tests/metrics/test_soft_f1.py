"""Tests for soft_f1 metric — AP-3, AP-15 compliance."""

import pytest

from glass.metrics.soft_f1 import SoftF1Metric
from tests.conftest import make_output, make_sample


class TestSoftF1:
    def test_perfect_match(self):
        m = SoftF1Metric()
        assert m.compute(make_output("Paris"), make_sample(gold="Paris")) == 1.0

    def test_partial_match_precision_biased(self):
        m = SoftF1Metric()
        # gold=["paris"], pred=["paris","is","nice"] → p=1/3, r=1.0, f1=0.5
        assert m.compute(make_output("Paris is nice"), make_sample(gold="Paris")) == 0.5

    def test_partial_match(self):
        m = SoftF1Metric()
        # gold=["paris","france"], pred=["paris","is","great"]
        # overlap=1, p=1/3, r=1/2 → f1=0.4
        assert abs(m.compute(make_output("Paris is great"), make_sample(gold="Paris France")) - 0.4) < 1e-6

    def test_no_match(self):
        m = SoftF1Metric()
        assert m.compute(make_output("London Berlin"), make_sample(gold="Paris France")) == 0.0

    def test_empty_gold_is_none(self):
        m = SoftF1Metric()
        assert m.compute(make_output("some answer"), make_sample(gold="")) is None

    def test_empty_prediction_is_zero(self):
        m = SoftF1Metric()
        assert m.compute(make_output(""), make_sample(gold="Paris")) == 0.0

    def test_returns_none_on_error(self):
        m = SoftF1Metric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_category_is_correctness(self):
        assert SoftF1Metric().category == "correctness"
