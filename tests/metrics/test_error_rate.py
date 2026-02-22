"""Tests for error_rate metric."""

from glass.metrics.error_rate import ErrorRateMetric
from tests.conftest import make_output, make_sample


class TestErrorRate:
    def test_clean_run(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(), make_sample()) == 0.0

    def test_flags_crash(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(error_type="crash"), make_sample()) == 1.0

    def test_flags_nonzero_exit(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(exit_code=1), make_sample()) == 1.0

    def test_flags_timeout(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(error_type="timeout"), make_sample()) == 1.0

    def test_flags_api_error(self):
        m = ErrorRateMetric()
        assert m.compute(make_output(error_type="api_error"), make_sample()) == 1.0

    def test_category_is_operational(self):
        assert ErrorRateMetric().category == "operational"
