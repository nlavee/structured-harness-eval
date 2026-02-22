"""Tests for citation_presence metric."""

from glass.metrics.citation_presence import CitationPresenceMetric
from tests.conftest import make_output, make_sample


class TestCitationPresence:
    def test_no_citation(self):
        m = CitationPresenceMetric()
        assert m.compute(make_output("Just a normal answer."), make_sample()) == 0.0

    def test_brackets_citation(self):
        m = CitationPresenceMetric()
        assert m.compute(make_output("Answer [2]"), make_sample()) == 1.0

    def test_doc_citation(self):
        m = CitationPresenceMetric()
        assert m.compute(make_output("Answer [doc 4]"), make_sample()) == 1.0
        assert m.compute(make_output("Answer [Document 12]"), make_sample()) == 1.0

    def test_source_citation(self):
        m = CitationPresenceMetric()
        assert m.compute(make_output("Answer. Source: 3"), make_sample()) == 1.0

    def test_returns_none_on_error(self):
        m = CitationPresenceMetric()
        assert m.compute(make_output(error_type="timeout"), make_sample()) is None

    def test_year_in_brackets_not_citation(self):
        """[2023] should NOT be detected as a citation (too many digits)."""
        m = CitationPresenceMetric()
        assert m.compute(make_output("Published in [2023]"), make_sample()) == 0.0

    def test_three_digit_is_citation(self):
        m = CitationPresenceMetric()
        assert m.compute(make_output("See reference [123]"), make_sample()) == 1.0
