"""Tests for judge_score metric (mocked — no API calls)."""

from unittest.mock import patch

from glass.judges.llm import LLMJudge
from glass.metrics.judge_score import JudgeScoreMetric
from tests.conftest import make_output, make_sample


class TestJudgeScore:
    def test_requires_judge_property(self):
        assert JudgeScoreMetric().requires_judge is True

    def test_category_is_correctness(self):
        assert JudgeScoreMetric().category == "correctness"

    def test_returns_none_on_error(self):
        m = JudgeScoreMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_judge_correct(self):
        with patch("glass.judges.llm.LLMJudge._call_llm") as mock_call:
            with patch("glass.judges.llm.nltk.download"):
                j = LLMJudge("openai", "gpt-4")
                mock_call.return_value = "CORRECT"
                score, _ = j.evaluate_correctness("q", "a", "p")
                assert score == 1.0

    def test_judge_incorrect(self):
        with patch("glass.judges.llm.LLMJudge._call_llm") as mock_call:
            with patch("glass.judges.llm.nltk.download"):
                j = LLMJudge("openai", "gpt-4")
                mock_call.return_value = "INCORRECT"
                score, _ = j.evaluate_correctness("q", "a", "p")
                assert score == 0.0
