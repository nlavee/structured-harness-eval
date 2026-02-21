"""Extended statistics tests: seeding, edge cases, statistics report generation."""

import json

import numpy as np
import pytest

from glass.statistics.bootstrap import compute_ci
from glass.statistics.effect_size import rank_biserial
from glass.statistics.significance import wilcoxon_test

# --------------------------------------------------------------------------- #
# Bootstrap: seeding (AP-23)                                                  #
# --------------------------------------------------------------------------- #


class TestBootstrapSeeding:
    def test_same_seed_same_result(self):
        data = [1.0] * 50 + [0.0] * 50
        r1 = compute_ci(data, n_resamples=1000, seed=99)
        r2 = compute_ci(data, n_resamples=1000, seed=99)
        assert r1 == r2

    def test_different_seed_may_differ(self):
        data = [1.0] * 50 + [0.0] * 50
        # With only 50% variance, different seeds usually produce slightly different CIs
        r1 = compute_ci(data, n_resamples=500, seed=1)
        r2 = compute_ci(data, n_resamples=500, seed=2)
        # They should be close but may not be identical
        assert abs(r1[0] - r2[0]) < 0.1  # Sane range

    def test_empty_data_returns_nan(self):
        low, high = compute_ci([], seed=42)
        assert np.isnan(low)
        assert np.isnan(high)

    def test_ci_bounds_order(self):
        data = [0.8, 0.9, 0.7, 0.85, 0.75]
        low, high = compute_ci(data, n_resamples=1000, seed=42)
        assert low < high

    def test_ci_within_data_range(self):
        data = [0.0] * 30 + [1.0] * 70  # Mean ~0.7
        low, high = compute_ci(data, n_resamples=2000, seed=42)
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0
        assert low <= 0.7 <= high  # Mean should be inside CI


# --------------------------------------------------------------------------- #
# Wilcoxon edge cases                                                         #
# --------------------------------------------------------------------------- #


class TestWilcoxon:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_test([1, 2], [1, 2, 3])

    def test_all_same_returns_one(self):
        x = [0.5] * 10
        y = [0.5] * 10
        assert wilcoxon_test(x, y) == 1.0

    def test_one_tailed_greater(self):
        x = [1.0] * 8
        y = [0.0] * 8
        p = wilcoxon_test(x, y, alternative="greater")
        assert p < 0.05

    def test_one_tailed_less(self):
        x = [0.0] * 8
        y = [1.0] * 8
        p = wilcoxon_test(x, y, alternative="less")
        assert p < 0.05

    def test_two_tailed_symmetric(self):
        x = [1.0] * 6
        y = [0.0] * 6
        p_greater = wilcoxon_test(x, y, alternative="greater")
        p_two = wilcoxon_test(x, y, alternative="two-sided")
        # Two-sided p ≈ 2 * one-sided p
        assert p_two >= p_greater


# --------------------------------------------------------------------------- #
# Rank-biserial edge cases                                                    #
# --------------------------------------------------------------------------- #


class TestRankBiserial:
    def test_all_tied_returns_zero(self):
        x = [1.0] * 5
        y = [1.0] * 5
        assert rank_biserial(x, y) == 0.0

    def test_range_is_valid(self):
        x = [0.7, 0.8, 0.6, 0.9]
        y = [0.3, 0.4, 0.5, 0.2]
        r = rank_biserial(x, y)
        assert -1.0 <= r <= 1.0

    def test_negative_effect(self):
        # x consistently below y → negative r
        x = [0.1, 0.2, 0.1]
        y = [0.9, 0.8, 0.9]
        r = rank_biserial(x, y)
        assert r < 0


# --------------------------------------------------------------------------- #
# Statistics report integration                                               #
# --------------------------------------------------------------------------- #


class TestStatisticsReport:
    def _make_results(self, n=10):
        from glass.judges.base import EvalResult

        results = []
        for i in range(n):
            for sname, score, hr in [
                ("structured_harness_claude", 0.8, 0.1),
                ("claude-code", 0.5, 0.3),
            ]:
                results.append(
                    EvalResult(
                        sample_id=str(i),
                        system_name=sname,
                        domain="Legal" if i % 2 == 0 else "Academic",
                        metrics={
                            "judge_score": score + (0.1 if i % 3 == 0 else 0.0),
                            "hallucination_rate": hr,
                            "exact_match": 1.0 if score > 0.6 else 0.0,
                            "latency_s": 2.5,
                        },
                        judge_outputs={"correctness": "CORRECT"},
                    )
                )
        return results

    def test_generates_statistics_json(self, tmp_path):
        from glass.config.schema import Config
        from glass.reports.statistics_report import generate_statistics_report

        config = Config(
            **{
                "experiment": {"name": "test", "seed": 42},
                "dataset": {"name": "aa_lcr"},
                "systems": [{"name": "s", "type": "stub"}],
                "metrics": ["judge_score", "hallucination_rate"],
                "judges": {
                    "strategy": "fixed",
                    "fixed": {"provider": "openai", "model": "gpt-4"},
                    "templates": {"correctness": "t", "hallucination": "t"},
                },
                "statistics": {
                    "bootstrap_resamples": 100,
                    "alpha": 0.05,
                    "primary_test": "wilcoxon_one_tailed",
                    "secondary_test": "wilcoxon_two_tailed",
                },
                "output": {"runs_dir": str(tmp_path)},
            }
        )

        results = self._make_results(n=10)
        out = tmp_path / "statistics.json"
        generate_statistics_report(results, config, out)

        assert out.exists()
        with open(out) as f:
            data = json.load(f)

        # Check required top-level keys
        assert "system_stats" in data
        assert "primary_hypothesis" in data
        assert "secondary_comparisons" in data
        assert "per_domain" in data
        assert "multiple_comparisons_note" in data

        # Check system stats contain CIs
        for sname in ["structured_harness_claude", "claude-code"]:
            assert sname in data["system_stats"]
            js = data["system_stats"][sname].get("judge_score", {})
            assert "ci_low" in js
            assert "ci_high" in js

    def test_empty_results_produces_error_key(self, tmp_path):
        from glass.config.schema import Config
        from glass.reports.statistics_report import generate_statistics_report

        config = Config(
            **{
                "experiment": {"name": "test", "seed": 42},
                "dataset": {"name": "aa_lcr"},
                "systems": [{"name": "s", "type": "stub"}],
                "metrics": [],
                "judges": {
                    "strategy": "fixed",
                    "fixed": {"provider": "openai", "model": "gpt-4"},
                    "templates": {"correctness": "t", "hallucination": "t"},
                },
                "statistics": {
                    "bootstrap_resamples": 100,
                    "alpha": 0.05,
                    "primary_test": "w",
                    "secondary_test": "t",
                },
                "output": {},
            }
        )

        out = tmp_path / "statistics.json"
        stats = generate_statistics_report([], config, out)
        assert "error" in stats
