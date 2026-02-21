"""
Error taxonomy and qualitative analysis module.

Supports the "why" section of the GLASS paper by categorising failures,
finding interesting divergence cases between systems, and producing
domain × error-type cross-tabulations.

Usage (called from CLI or notebook):
    from glass.analysis.error_taxonomy import ErrorAnalyser
    analyser = ErrorAnalyser(results, raw_outputs)
    report = analyser.run()
    analyser.write_report(report, output_path)
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from glass.judges.base import EvalResult
from glass.systems.base import RawOutput

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Taxonomy                                                                     #
# --------------------------------------------------------------------------- #

ERROR_TYPE_LABELS = {
    "timeout": "Timeout",
    "api_error": "API Error",
    "refusal": "Refusal",
    "malformed": "Malformed Output",
    "crash": "CLI Crash",
    None: "No Error",
}

CORRECTNESS_METRIC = "judge_score"
HALLUCINATION_METRIC = "hallucination_rate"

# Threshold below which judge_score is "correct"
CORRECT_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Data helpers                                                                 #
# --------------------------------------------------------------------------- #


def _index_raw(raw_outputs: List[RawOutput]) -> Dict[Tuple[str, str], RawOutput]:
    return {(r.system_name, r.sample_id): r for r in raw_outputs}


def _index_results(results: List[EvalResult]) -> Dict[Tuple[str, str], EvalResult]:
    return {(r.system_name, r.sample_id): r for r in results}


# --------------------------------------------------------------------------- #
# Main analyser                                                                #
# --------------------------------------------------------------------------- #


class ErrorAnalyser:
    """Produces a structured error analysis report from completed eval results."""

    def __init__(
        self,
        results: List[EvalResult],
        raw_outputs: Optional[List[RawOutput]] = None,
    ):
        self.results = results
        self.raw_idx = _index_raw(raw_outputs) if raw_outputs else {}
        self.result_idx = _index_results(results)
        self.systems = sorted({r.system_name for r in results})
        self.domains = sorted({r.domain for r in results})
        self.sample_ids = sorted({r.sample_id for r in results})

    # ------------------------------------------------------------------ #
    # 1. Error-type × domain cross-tabulation                             #
    # ------------------------------------------------------------------ #

    def error_type_crosstab(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Returns {system: {domain: {error_type: count}}}."""
        table: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: Counter()))
        for (sname, sid), raw in self.raw_idx.items():
            result = self.result_idx.get((sname, sid))
            if result is None:
                continue
            label = ERROR_TYPE_LABELS.get(raw.error_type, str(raw.error_type))
            table[sname][result.domain][label] += 1
        return {s: {d: dict(c) for d, c in dom.items()} for s, dom in table.items()}

    # ------------------------------------------------------------------ #
    # 2. Divergence: structured harness succeeds, baseline fails (and vice versa) #
    # ------------------------------------------------------------------ #

    def divergence_cases(
        self,
        harness_system: str,
        baseline_system: str,
        metric: str = CORRECTNESS_METRIC,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find samples where one system succeeds and the other fails."""
        harness_wins: List[Dict[str, Any]] = []
        base_wins: List[Dict[str, Any]] = []

        for sid in self.sample_ids:
            harness = self.result_idx.get((harness_system, sid))
            base = self.result_idx.get((baseline_system, sid))
            if harness is None or base is None:
                continue

            harness_val = harness.metrics.get(metric)
            base_val = base.metrics.get(metric)
            if harness_val is None or base_val is None:
                continue

            harness_ok = harness_val >= CORRECT_THRESHOLD
            base_ok = base_val >= CORRECT_THRESHOLD

            if harness_ok and not base_ok:
                harness_wins.append(
                    {
                        "sample_id": sid,
                        "domain": harness.domain,
                        f"{harness_system}_{metric}": harness_val,
                        f"{baseline_system}_{metric}": base_val,
                    }
                )
            elif base_ok and not harness_ok:
                base_wins.append(
                    {
                        "sample_id": sid,
                        "domain": harness.domain,
                        f"{harness_system}_{metric}": harness_val,
                        f"{baseline_system}_{metric}": base_val,
                    }
                )

        return {
            f"{harness_system}_wins": harness_wins,
            f"{baseline_system}_wins": base_wins,
        }

    # ------------------------------------------------------------------ #
    # 3. Verbosity vs correctness scatter data                            #
    # ------------------------------------------------------------------ #

    def verbosity_vs_correctness(
        self,
        verbosity_metric: str = "verbosity",
        correctness_metric: str = CORRECTNESS_METRIC,
    ) -> List[Dict[str, Any]]:
        """Return per-(sample, system) rows for scatter analysis."""
        rows = []
        for r in self.results:
            v = r.metrics.get(verbosity_metric)
            c = r.metrics.get(correctness_metric)
            if v is not None and c is not None:
                rows.append(
                    {
                        "sample_id": r.sample_id,
                        "system_name": r.system_name,
                        "domain": r.domain,
                        verbosity_metric: v,
                        correctness_metric: c,
                    }
                )
        return rows

    # ------------------------------------------------------------------ #
    # 4. Hallucination by domain                                          #
    # ------------------------------------------------------------------ #

    def hallucination_by_domain(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Returns {domain: {system: mean_hallucination_rate}}."""
        buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            hr = r.metrics.get(HALLUCINATION_METRIC)
            if hr is not None:
                buckets[r.domain][r.system_name].append(hr)
        return {
            domain: {sname: (sum(vals) / len(vals) if vals else None) for sname, vals in sys_vals.items()}
            for domain, sys_vals in buckets.items()
        }

    # ------------------------------------------------------------------ #
    # 5. Summary report                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """Produce the full error analysis report dict."""
        report: Dict[str, Any] = {
            "systems": self.systems,
            "domains": self.domains,
            "n_samples": len(self.sample_ids),
            "error_type_crosstab": self.error_type_crosstab(),
            "hallucination_by_domain": self.hallucination_by_domain(),
            "verbosity_vs_correctness": self.verbosity_vs_correctness(),
            "divergence": {},
        }

        # Compute divergence for each structured_harness × baseline pair
        harness_systems = [s for s in self.systems if "structured_harness" in s.lower()]
        base_systems = [s for s in self.systems if s in {"claude-code", "gemini-cli", "codex-cli"}]
        for harness in harness_systems:
            for base in base_systems:
                report["divergence"][f"{harness}_vs_{base}"] = self.divergence_cases(harness, base)

        return report

    def write_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Write report as JSON (machine-readable) and append Markdown to summary."""
        json_path = Path(output_path).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("[GLASS] Saved error analysis → %s", json_path)

        # Append Markdown narrative
        md_path = Path(output_path).with_suffix(".md")
        lines = ["# Error Analysis\n"]

        # Error type summary table
        lines.append("## Error Types by System\n")
        for sname, domain_data in report.get("error_type_crosstab", {}).items():
            lines.append(f"### {sname}\n")
            for domain, counts in sorted(domain_data.items()):
                lines.append(f"- **{domain}**: {counts}\n")

        # Hallucination by domain
        lines.append("\n## Hallucination Rate by Domain\n")
        hr_table: Dict[str, Any] = report.get("hallucination_by_domain", {})
        for domain, sys_vals in sorted(hr_table.items()):
            parts = ", ".join(f"{s}: {v:.3f}" if v is not None else f"{s}: N/A" for s, v in sys_vals.items())
            lines.append(f"- **{domain}**: {parts}\n")

        # Divergence cases
        lines.append("\n## Divergence Cases\n")
        for comparison, cases in report.get("divergence", {}).items():
            lines.append(f"### {comparison}\n")
            for case_label, case_list in cases.items():
                lines.append(f"- **{case_label}**: {len(case_list)} samples\n")
                for case in case_list[:5]:  # Show first 5 examples
                    lines.append(f"  - sample {case['sample_id']} (domain: {case['domain']})\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info("[GLASS] Saved error analysis markdown → %s", md_path)
