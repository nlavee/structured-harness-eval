"""Tests for latency_s metric — AP-9 compliance."""

from glass.metrics.latency import LatencyMetric
from tests.conftest import make_output, make_sample


class TestLatency:
    def test_returns_latency(self):
        m = LatencyMetric()
        assert m.compute(make_output(latency=5.0), make_sample()) == 5.0

    def test_returns_latency_on_timeout(self):
        """Latency is always valid — even on errors (AP-9)."""
        m = LatencyMetric()
        assert m.compute(make_output(error_type="timeout", latency=600.0), make_sample()) == 600.0

    def test_returns_latency_on_crash(self):
        m = LatencyMetric()
        assert m.compute(make_output(error_type="crash", latency=3.2), make_sample()) == 3.2

    def test_returns_latency_on_api_error(self):
        m = LatencyMetric()
        assert m.compute(make_output(error_type="api_error", latency=0.5), make_sample()) == 0.5

    def test_category_is_operational(self):
        assert LatencyMetric().category == "operational"
