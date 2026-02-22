"""Tests for exact_match metric — AP-3, AP-14 compliance."""

from glass.metrics.exact_match import ExactMatchMetric
from tests.conftest import make_output, make_sample


class TestExactMatch:
    def test_case_insensitive_match(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("paris"), make_sample(gold="Paris")) == 1.0

    def test_case_insensitive_upper(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("PARIS"), make_sample(gold="paris")) == 1.0

    def test_mismatch(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("London"), make_sample(gold="Paris")) == 0.0

    def test_trailing_punctuation(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("Paris."), make_sample(gold="Paris")) == 1.0

    def test_extra_whitespace(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("  Paris  "), make_sample(gold="Paris")) == 1.0

    def test_unicode_normalisation(self):
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        m = ExactMatchMetric()
        assert m.compute(make_output(decomposed), make_sample(gold=composed)) == 1.0

    def test_returns_none_on_error(self):
        m = ExactMatchMetric()
        assert m.compute(make_output(error_type="timeout"), make_sample()) is None

    def test_category_is_correctness(self):
        assert ExactMatchMetric().category == "correctness"
