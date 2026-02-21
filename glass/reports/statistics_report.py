"""
Phase 3: Statistics computation and statistics.json generation.

Implements PLAN.md Phase 3 requirements:
- Bootstrap 95% CI for every metric × SUT  (AP-17: no means without CIs)
- Wilcoxon signed-rank for paired comparisons  (AP-18: no t-test on binary scores)
  * One-tailed for primary metrics (judge_score, hallucination_rate)
  * Two-tailed for secondary metrics
- Rank-biserial effect size for all comparisons
- Per-domain descriptive stats (AP-19: no significance claims at N≈14)
- Multiple-comparisons acknowledgment metadata (AP-20)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from glass.config.schema import Config
from glass.judges.base import EvalResult
from glass.statistics.bootstrap import compute_ci
from glass.statistics.effect_size import rank_biserial
from glass.statistics.significance import wilcoxon_test

logger = logging.getLogger(__name__)

# Primary metrics: one-tailed tests (per PLAN.md pre-specification)
# Direction for judge_score: structured_harness ≥ baseline → "greater"
# Direction for hallucination_rate: structured_harness ≤ baseline → "less"
PRIMARY_METRICS = {"judge_score", "hallucination_rate"}
PRIMARY_DIRECTIONS: Dict[str, str] = {
    "judge_score": "greater",
    "hallucination_rate": "less",
}

# Baselines to compare Structured Harness against (PLAN.md core scientific contribution)
BASELINE_SYSTEMS = {"claude-code", "gemini-cli", "codex-cli"}
HARNESS_PREFIX = "structured_harness"


def _group_by_system(results: List[EvalResult]) -> Dict[str, List[EvalResult]]:
    grouped: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in results:
        grouped[r.system_name].append(r)
    return dict(grouped)


def _get_paired_values(
    results_a: List[EvalResult],
    results_b: List[EvalResult],
    metric: str,
) -> tuple[List[float], List[float]]:
    """Extract paired (a, b) metric values for samples present in both systems."""
    map_a = {r.sample_id: r.metrics.get(metric) for r in results_a}
    map_b = {r.sample_id: r.metrics.get(metric) for r in results_b}
    common_ids = [sid for sid in map_a if sid in map_b]

    pairs = [(map_a[sid], map_b[sid]) for sid in common_ids if map_a[sid] is not None and map_b[sid] is not None]
    if not pairs:
        return [], []
    x, y = zip(*pairs)
    return list(x), list(y)


def _aggregate_metrics(
    results: List[EvalResult],
    metric_names: List[str],
    n_resamples: int,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    """Compute mean, std, median, and bootstrap CI per metric."""
    agg: Dict[str, Any] = {}
    for metric in metric_names:
        vals = [r.metrics.get(metric) for r in results if r.metrics.get(metric) is not None]
        if not vals:
            agg[metric] = {
                "n": 0,
                "mean": None,
                "std": None,
                "median": None,
                "ci_low": None,
                "ci_high": None,
            }
            continue
        arr = np.array(vals, dtype=float)
        ci_low, ci_high = compute_ci(vals, n_resamples=n_resamples, alpha=alpha, seed=seed)
        agg[metric] = {
            "n": len(vals),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return agg


def generate_statistics_report(
    results: List[EvalResult],
    config: Config,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Compute and write statistics.json.

    Returns the stats dict so callers (e.g. summary.py) can embed highlights.
    """
    if not results:
        stats: Dict[str, Any] = {"error": "No results to analyse."}
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    sc = config.statistics
    seed = config.experiment.seed
    n_resamples = sc.bootstrap_resamples
    alpha = sc.alpha

    grouped = _group_by_system(results)
    system_names = sorted(grouped.keys())

    # Discover all metric names across results
    all_metrics: List[str] = sorted({k for r in results for k in r.metrics})
    secondary_metrics = [m for m in all_metrics if m not in PRIMARY_METRICS]

    # -----------------------------------------------------------
    # 1. Per-system aggregate stats
    # -----------------------------------------------------------
    system_stats: Dict[str, Any] = {}
    for sname, sresults in grouped.items():
        system_stats[sname] = _aggregate_metrics(sresults, all_metrics, n_resamples=n_resamples, alpha=alpha, seed=seed)

    # -----------------------------------------------------------
    # 2. Primary hypothesis: structured harness vs each baseline (pre-specified)
    # -----------------------------------------------------------
    primary_hypothesis: Dict[str, Any] = {}
    harness_systems = [s for s in system_names if HARNESS_PREFIX in s.lower()]
    baseline_systems = [s for s in system_names if s in BASELINE_SYSTEMS]

    for harness_sys in harness_systems:
        for base_sys in baseline_systems:
            key = f"{harness_sys}_vs_{base_sys}"
            primary_hypothesis[key] = {}
            for metric in PRIMARY_METRICS:
                if metric not in all_metrics:
                    continue
                x, y = _get_paired_values(grouped[harness_sys], grouped[base_sys], metric)
                if len(x) < 4:  # Wilcoxon requires at least 4 pairs to be meaningful
                    primary_hypothesis[key][metric] = {
                        "n_pairs": len(x),
                        "note": "Insufficient paired samples for Wilcoxon test",
                    }
                    continue
                direction = PRIMARY_DIRECTIONS.get(metric, "greater")
                p = wilcoxon_test(x, y, alternative=direction)
                r = rank_biserial(x, y)
                primary_hypothesis[key][metric] = {
                    "n_pairs": len(x),
                    "p_value": p,
                    "effect_size": r,
                    "alternative": direction,
                    "significant_at_alpha": p < alpha,
                }

    # -----------------------------------------------------------
    # 3. Secondary metrics: two-tailed Wilcoxon for all system pairs
    # -----------------------------------------------------------
    secondary_comparisons: Dict[str, Any] = {}
    for i, sys_a in enumerate(system_names):
        for sys_b in system_names[i + 1 :]:
            key = f"{sys_a}_vs_{sys_b}"
            secondary_comparisons[key] = {}
            for metric in secondary_metrics:
                x, y = _get_paired_values(grouped[sys_a], grouped[sys_b], metric)
                if len(x) < 4:
                    secondary_comparisons[key][metric] = {
                        "n_pairs": len(x),
                        "note": "Insufficient paired samples",
                    }
                    continue
                p = wilcoxon_test(x, y, alternative="two-sided")
                r = rank_biserial(x, y)
                secondary_comparisons[key][metric] = {
                    "n_pairs": len(x),
                    "p_value": p,
                    "effect_size": r,
                    "significant_at_alpha": p < alpha,
                }

    # -----------------------------------------------------------
    # 4. Per-domain descriptive stats (AP-19: no significance testing)
    # -----------------------------------------------------------
    domains = sorted({r.domain for r in results})
    per_domain: Dict[str, Any] = {}
    for domain in domains:
        domain_results = [r for r in results if r.domain == domain]
        domain_grouped = _group_by_system(domain_results)
        per_domain[domain] = {}
        for sname, sresults in domain_grouped.items():
            domain_vals: Dict[str, Any] = {}
            for metric in all_metrics:
                vals = [r.metrics.get(metric) for r in sresults if r.metrics.get(metric) is not None]
                domain_vals[metric] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else None,
                }
            per_domain[domain][sname] = domain_vals

    # -----------------------------------------------------------
    # 5. Multiple-comparisons metadata (AP-20 acknowledgment)
    # -----------------------------------------------------------
    n_systems = len(system_names)
    n_metrics = len(all_metrics)
    n_comparisons = (n_systems * (n_systems - 1) // 2) * n_metrics
    mc_note = (
        f"This run compared {n_systems} systems across {n_metrics} metrics "
        f"({n_comparisons} total pairwise comparisons at α={alpha}). "
        f"Expected false positives by chance: ~{n_comparisons * alpha:.1f}. "
        "Primary claims are pre-specified (judge_score, hallucination_rate, "
        "Structured Harness vs baselines only). All other comparisons are exploratory."
    )

    stats = {
        "run_config": {
            "n_samples": len({r.sample_id for r in results}),
            "n_systems": n_systems,
            "systems": system_names,
            "metrics": all_metrics,
            "bootstrap_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
        },
        "system_stats": system_stats,
        "primary_hypothesis": primary_hypothesis,
        "secondary_comparisons": secondary_comparisons,
        "per_domain": per_domain,
        "multiple_comparisons_note": mc_note,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info("[GLASS] Saved statistics → %s", output_path)
    return stats
