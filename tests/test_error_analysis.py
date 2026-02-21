"""Tests for the error analysis / error taxonomy module."""

from glass.analysis.error_taxonomy import ErrorAnalyser
from glass.judges.base import EvalResult
from glass.systems.base import RawOutput


def make_result(sample_id, system, score=1.0, hr=0.0, domain="Legal"):
    return EvalResult(
        sample_id=sample_id,
        system_name=system,
        domain=domain,
        metrics={"judge_score": score, "hallucination_rate": hr, "verbosity": 1.2},
        judge_outputs={},
    )


def make_raw(sample_id, system, error_type=None):
    return RawOutput(
        sample_id=sample_id,
        system_name=system,
        command=["stub"],
        prompt="",
        output="answer",
        latency_s=1.0,
        exit_code=0 if error_type is None else 1,
        stderr="",
        error_type=error_type,
        timestamp="t",
    )


class TestErrorAnalyser:
    def _make_data(self):
        results = [
            make_result("1", "structured_harness_claude", score=1.0, domain="Legal"),
            make_result("1", "claude-code", score=0.0, domain="Legal"),
            make_result("2", "structured_harness_claude", score=0.0, domain="Academic"),
            make_result("2", "claude-code", score=1.0, domain="Academic"),
            make_result("3", "structured_harness_claude", score=1.0, domain="Legal"),
            make_result("3", "claude-code", score=0.0, domain="Legal"),
        ]
        raw = [
            make_raw("1", "structured_harness_claude"),
            make_raw("1", "claude-code", error_type="timeout"),
            make_raw("2", "structured_harness_claude"),
            make_raw("2", "claude-code"),
            make_raw("3", "structured_harness_claude"),
            make_raw("3", "claude-code", error_type="crash"),
        ]
        return results, raw

    def test_error_type_crosstab_counts(self):
        results, raw = self._make_data()
        analyser = ErrorAnalyser(results, raw)
        crosstab = analyser.error_type_crosstab()

        # claude-code had 2 errors (timeout + crash) across Legal domain samples
        claude_data = crosstab.get("claude-code", {})
        # Sample 1 Legal = Timeout, Sample 3 Legal = Crash
        legal = claude_data.get("Legal", {})
        assert legal.get("Timeout", 0) + legal.get("CLI Crash", 0) == 2

    def test_divergence_finds_hw_wins(self):
        results, raw = self._make_data()
        analyser = ErrorAnalyser(results, raw)
        div = analyser.divergence_cases("structured_harness_claude", "claude-code")

        hw_wins = div["structured_harness_claude_wins"]
        base_wins = div["claude-code_wins"]

        # Samples 1 and 3: structured harness succeeds, claude-code fails
        assert len(hw_wins) == 2
        # Sample 2: claude-code succeeds, structured harness fails
        assert len(base_wins) == 1

    def test_hallucination_by_domain(self):
        results, raw = self._make_data()
        analyser = ErrorAnalyser(results, raw)
        by_domain = analyser.hallucination_by_domain()

        assert "Legal" in by_domain
        assert "structured_harness_claude" in by_domain["Legal"]

    def test_verbosity_vs_correctness_rows(self):
        results, raw = self._make_data()
        analyser = ErrorAnalyser(results, raw)
        rows = analyser.verbosity_vs_correctness()

        # 6 results, each has verbosity and judge_score
        assert len(rows) == 6
        for row in rows:
            assert "verbosity" in row
            assert "judge_score" in row

    def test_run_produces_all_keys(self):
        results, raw = self._make_data()
        analyser = ErrorAnalyser(results, raw)
        report = analyser.run()

        assert "error_type_crosstab" in report
        assert "hallucination_by_domain" in report
        assert "verbosity_vs_correctness" in report
        assert "divergence" in report

    def test_write_report(self, tmp_path):
        results, raw = self._make_data()
        analyser = ErrorAnalyser(results, raw)
        report = analyser.run()
        analyser.write_report(report, tmp_path / "error_analysis")

        assert (tmp_path / "error_analysis.json").exists()
        assert (tmp_path / "error_analysis.md").exists()
