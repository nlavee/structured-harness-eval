"""Tests for answer_length metric."""

from glass.metrics.answer_length import AnswerLengthMetric
from tests.conftest import make_output, make_sample


class TestAnswerLength:
    def test_empty_answer(self):
        m = AnswerLengthMetric()
        assert m.compute(make_output(""), make_sample()) == 0.0

    def test_normal_answer(self):
        m = AnswerLengthMetric()
        assert m.compute(make_output("This has five words total"), make_sample()) == 5.0

    def test_single_word(self):
        m = AnswerLengthMetric()
        assert m.compute(make_output("Paris"), make_sample()) == 1.0

    def test_returns_none_on_error(self):
        m = AnswerLengthMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_category_is_behavioral(self):
        assert AnswerLengthMetric().category == "behavioral"
