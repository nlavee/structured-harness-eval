"""Edge case tests for all metrics — ensures AP-3, AP-14, AP-15, AP-16 compliance."""

from glass.datasets.base import EvaluationSample
from glass.metrics.error_rate import ErrorRateMetric
from glass.metrics.exact_match import ExactMatchMetric
from glass.metrics.latency import LatencyMetric
from glass.metrics.refusal import RefusalMetric
from glass.metrics.soft_recall import SoftRecallMetric
from glass.metrics.verbosity import VerbosityMetric
from glass.systems.base import RawOutput


def make_output(
    output_text="answer",
    error_type=None,
    exit_code=0,
    latency=1.0,
):
    return RawOutput(
        sample_id="x",
        system_name="s",
        command=["stub"],
        prompt="",
        output=output_text,
        latency_s=latency,
        exit_code=exit_code,
        stderr="",
        error_type=error_type,
        timestamp="t",
    )


def make_sample(gold="answer", question="q"):
    return EvaluationSample(
        sample_id="x",
        domain="d",
        question=question,
        gold_answer=gold,
        context_prompt="ctx",
        input_tokens=10,
        metadata={},
    )


# --------------------------------------------------------------------------- #
# AP-3: Metric failure → None, never 0.0                                      #
# --------------------------------------------------------------------------- #


class TestMetricsReturnNoneOnError:
    def test_exact_match_none_on_error(self):
        m = ExactMatchMetric()
        assert m.compute(make_output(error_type="timeout"), make_sample()) is None

    def test_soft_recall_none_on_error(self):
        m = SoftRecallMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) is None

    def test_verbosity_none_on_error(self):
        m = VerbosityMetric()
        assert m.compute(make_output(error_type="api_error"), make_sample()) is None

    def test_refusal_handles_error_type(self):
        """A 'refusal' error_type should directly return 1.0, not None."""
        m = RefusalMetric()
        assert m.compute(make_output(error_type="refusal"), make_sample()) == 1.0

    def test_error_rate_flags_crash(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) == 1.0

    def test_error_rate_flags_nonzero_exit(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(exit_code=1), make_sample()) == 1.0

    def test_error_rate_ok_when_clean(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(), make_sample()) == 0.0


# --------------------------------------------------------------------------- #
# AP-14: Exact match normalisation                                             #
# --------------------------------------------------------------------------- #


class TestExactMatchNormalisation:
    def test_case_insensitive(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("PARIS"), make_sample(gold="paris")) == 1.0

    def test_trailing_punctuation(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("Paris."), make_sample(gold="Paris")) == 1.0

    def test_extra_whitespace(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("  Paris  "), make_sample(gold="Paris")) == 1.0

    def test_unicode_normalisation(self):
        # é composed vs decomposed
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        m = ExactMatchMetric()
        assert m.compute(make_output(decomposed), make_sample(gold=composed)) == 1.0

    def test_mismatch(self):
        m = ExactMatchMetric()
        assert m.compute(make_output("London"), make_sample(gold="Paris")) == 0.0


# --------------------------------------------------------------------------- #
# AP-15: Undefined metric on empty/missing gold                               #
# --------------------------------------------------------------------------- #


class TestUndefinedCases:
    def test_soft_recall_empty_gold_is_none(self):
        """Empty gold = undefined recall (AP-15 style: don't return 0.0)."""
        m = SoftRecallMetric()
        result = m.compute(make_output("some answer"), make_sample(gold=""))
        assert result is None

    def test_verbosity_empty_gold_is_none(self):
        m = VerbosityMetric()
        result = m.compute(make_output("some answer"), make_sample(gold=""))
        assert result is None

    def test_soft_recall_empty_prediction(self):
        """Empty prediction gives 0.0 recall (no tokens matched)."""
        m = SoftRecallMetric()
        result = m.compute(make_output(""), make_sample(gold="Paris"))
        assert result == 0.0


# --------------------------------------------------------------------------- #
# Soft recall computation                                                      #
# --------------------------------------------------------------------------- #


class TestSoftRecall:
    def test_perfect_match(self):
        m = SoftRecallMetric()
        assert m.compute(make_output("Paris"), make_sample(gold="Paris")) == 1.0

    def test_partial_match(self):
        m = SoftRecallMetric()
        # gold = "paris france" (2 tokens); pred has "paris" only → recall = 0.5
        assert m.compute(make_output("Paris is great"), make_sample(gold="Paris France")) == 0.5

    def test_no_match(self):
        m = SoftRecallMetric()
        assert m.compute(make_output("London Berlin"), make_sample(gold="Paris France")) == 0.0


# --------------------------------------------------------------------------- #
# Verbosity                                                                   #
# --------------------------------------------------------------------------- #


class TestVerbosity:
    def test_equal_length(self):
        m = VerbosityMetric()
        assert m.compute(make_output("abc"), make_sample(gold="xyz")) == 1.0

    def test_longer_prediction(self):
        m = VerbosityMetric()
        # "12345" / "123" = 5/3 ≈ 1.667
        result = m.compute(make_output("12345"), make_sample(gold="123"))
        assert abs(result - 5 / 3) < 1e-6


# --------------------------------------------------------------------------- #
# Refusal detection                                                           #
# --------------------------------------------------------------------------- #


class TestRefusal:
    def test_no_refusal(self):
        m = RefusalMetric()
        assert m.compute(make_output("The answer is 42."), make_sample()) == 0.0

    def test_refusal_pattern_i_cannot_answer(self):
        m = RefusalMetric()
        assert m.compute(make_output("I cannot answer this question."), make_sample()) == 1.0

    def test_refusal_pattern_case_insensitive(self):
        m = RefusalMetric()
        assert m.compute(make_output("i cannot provide that information"), make_sample()) == 1.0

    def test_explicit_error_type_refusal(self):
        m = RefusalMetric()
        assert m.compute(make_output(error_type="refusal"), make_sample()) == 1.0


# --------------------------------------------------------------------------- #
# Latency                                                                     #
# --------------------------------------------------------------------------- #


class TestLatency:
    def test_returns_latency(self):
        m = LatencyMetric()
        assert m.compute(make_output(latency=3.14), make_sample()) == 3.14
