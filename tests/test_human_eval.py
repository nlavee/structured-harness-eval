"""Tests for human evaluation module: exporter, importer, and Cohen's Kappa."""

import csv
from unittest.mock import MagicMock

import pytest

from glass.human_eval.agreement import cohens_kappa, compute_agreement
from glass.human_eval.importer import ImportValidationError, import_human_labels
from glass.judges.base import EvalResult

# --------------------------------------------------------------------------- #
# Cohen's Kappa                                                               #
# --------------------------------------------------------------------------- #


class TestCohensKappa:
    def test_perfect_agreement(self):
        h = [1, 0, 1, 1, 0]
        a = [1, 0, 1, 1, 0]
        assert cohens_kappa(h, a) == pytest.approx(1.0)

    def test_zero_agreement_chance_level(self):
        # Perfectly anti-correlated given balanced base rate → κ = -1
        h = [1, 1, 0, 0]
        a = [0, 0, 1, 1]
        k = cohens_kappa(h, a)
        assert k < 0

    def test_empty_lists(self):
        import math

        assert math.isnan(cohens_kappa([], []))

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            cohens_kappa([1, 0], [1, 0, 1])

    def test_degenerate_all_same_label(self):
        import math

        h = [1, 1, 1, 1]
        a = [1, 1, 1, 1]
        # p_expected = 1.0, denominator = 0 → nan
        assert math.isnan(cohens_kappa(h, a))

    def test_moderate_agreement(self):
        # 3 agree out of 4 → p_o = 0.75; with balanced labels p_e ≈ 0.5
        h = [1, 0, 1, 0]
        a = [1, 0, 1, 1]  # Last one differs
        k = cohens_kappa(h, a)
        assert 0 < k < 1


# --------------------------------------------------------------------------- #
# compute_agreement                                                           #
# --------------------------------------------------------------------------- #


def make_result(sample_id, system, score, human_label=None, domain="Legal"):
    return EvalResult(
        sample_id=sample_id,
        system_name=system,
        domain=domain,
        metrics={"judge_score": score},
        judge_outputs={},
        human_label=human_label,
    )


class TestComputeAgreement:
    def test_only_labelled_samples_included(self):
        results = [
            make_result("1", "sys", 1.0, human_label=1),
            make_result("2", "sys", 0.0, human_label=None),  # unlabelled → skip
            make_result("3", "sys", 0.0, human_label=0),
        ]
        agreement = compute_agreement(results)
        assert agreement["overall"]["n"] == 2

    def test_per_system_kappa(self):
        results = [
            make_result("1", "sys_a", 1.0, human_label=1),
            make_result("2", "sys_a", 1.0, human_label=1),
            make_result("3", "sys_a", 0.0, human_label=0),
            make_result("4", "sys_a", 0.0, human_label=0),
        ]
        agreement = compute_agreement(results)
        assert "sys_a" in agreement["per_system"]
        assert agreement["per_system"]["sys_a"]["kappa"] == pytest.approx(1.0)

    def test_no_labelled_results(self):
        import math

        results = [make_result("1", "sys", 0.5, human_label=None)]
        agreement = compute_agreement(results)
        assert math.isnan(agreement["overall"]["kappa"])
        assert agreement["overall"]["n"] == 0


# --------------------------------------------------------------------------- #
# Importer validation                                                         #
# --------------------------------------------------------------------------- #


class TestImporter:
    def _write_csv(self, tmp_path, rows, fieldnames=None):
        if fieldnames is None:
            fieldnames = ["sample_id", "system_name", "human_label"]
        p = tmp_path / "labels.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_valid_import(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path,
            [
                {"sample_id": "1", "system_name": "sys", "human_label": "1"},
                {"sample_id": "2", "system_name": "sys", "human_label": "0"},
            ],
        )
        results = [
            make_result("1", "sys", 1.0),
            make_result("2", "sys", 0.0),
        ]
        store = MagicMock()
        updated, label_map = import_human_labels(csv_path, results, store)
        assert label_map == {"1": 1, "2": 0}
        assert results[0].human_label == 1
        assert results[1].human_label == 0

    def test_blank_label_raises(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path,
            [
                {"sample_id": "1", "system_name": "sys", "human_label": ""},
            ],
        )
        with pytest.raises(ImportValidationError, match="blank"):
            import_human_labels(csv_path, [], MagicMock())

    def test_invalid_label_value_raises(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path,
            [
                {"sample_id": "1", "system_name": "sys", "human_label": "2"},
            ],
        )
        with pytest.raises(ImportValidationError, match="0 or 1"):
            import_human_labels(csv_path, [], MagicMock())

    def test_missing_human_label_column_raises(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path,
            [{"sample_id": "1", "system_name": "sys"}],
            fieldnames=["sample_id", "system_name"],
        )
        with pytest.raises(ImportValidationError, match="human_label"):
            import_human_labels(csv_path, [], MagicMock())

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_human_labels(tmp_path / "nonexistent.csv", [], MagicMock())
