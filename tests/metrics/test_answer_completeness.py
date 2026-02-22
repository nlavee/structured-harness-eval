"""Tests for answer_completeness metric."""

from glass.metrics.answer_completeness import AnswerCompletenessMetric
from tests.conftest import make_output, make_sample


class TestAnswerCompleteness:
    def test_returns_none_on_error(self):
        m = AnswerCompletenessMetric()
        assert m.compute(make_output(error_type="timeout"), make_sample()) is None

    def test_empty_output(self):
        m = AnswerCompletenessMetric()
        assert m.compute(make_output(""), make_sample()) == 0.0

    def test_single_question_answered(self):
        m = AnswerCompletenessMetric()
        result = m.compute(make_output("The answer is Paris."), make_sample(question="What is the capital?"))
        assert result == 1.0

    def test_multi_question_partial(self):
        m = AnswerCompletenessMetric()
        result = m.compute(
            make_output("Paris."),
            make_sample(question="What is the capital? What is the population? What is the area?"),
        )
        assert result < 1.0

    def test_multi_question_fully_answered(self):
        m = AnswerCompletenessMetric()
        result = m.compute(
            make_output("Paris is the capital. The population is 2M. The area is large."),
            make_sample(question="What is the capital? What is the population? What is the area?"),
        )
        assert result == 1.0

    def test_capped_at_one(self):
        m = AnswerCompletenessMetric()
        result = m.compute(
            make_output("First. Second. Third. Fourth. Fifth."),
            make_sample(question="What?"),
        )
        assert result == 1.0

    def test_category_is_correctness(self):
        assert AnswerCompletenessMetric().category == "correctness"
