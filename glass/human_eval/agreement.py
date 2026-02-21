"""
Inter-rater agreement: Cohen's Kappa between human and automated judge.

Computes kappa per automated judge (system) and reports disagreement rows
for qualitative review, per PLAN.md Phase 5.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from glass.judges.base import EvalResult

logger = logging.getLogger(__name__)

CORRECTNESS_METRIC = "judge_score"
CORRECT_THRESHOLD = 0.5


def _binarise(value: float, threshold: float = CORRECT_THRESHOLD) -> int:
    return 1 if value >= threshold else 0


def cohens_kappa(
    human_labels: List[int],
    auto_labels: List[int],
) -> float:
    """
    Compute Cohen's Kappa for binary labels.

    κ = (p_o - p_e) / (1 - p_e)

    Returns κ in [-1, 1]; 1.0 means perfect agreement, 0.0 = chance level.
    Returns float("nan") if agreement cannot be computed (e.g. all same label).
    """
    if len(human_labels) != len(auto_labels):
        raise ValueError("human_labels and auto_labels must have the same length.")
    n = len(human_labels)
    if n == 0:
        return float("nan")

    h = np.array(human_labels, dtype=int)
    a = np.array(auto_labels, dtype=int)

    p_observed = float(np.mean(h == a))

    p_h1 = float(np.mean(h == 1))
    p_a1 = float(np.mean(a == 1))
    p_expected = p_h1 * p_a1 + (1 - p_h1) * (1 - p_a1)

    if abs(1 - p_expected) < 1e-10:
        return float("nan")  # Degenerate distribution

    return (p_observed - p_expected) / (1 - p_expected)


def compute_agreement(
    results: List[EvalResult],
    metric: str = CORRECTNESS_METRIC,
) -> Dict[str, Any]:
    """
    Compute Cohen's Kappa between human and automated judge per system.

    Only samples with non-None human_label are included.

    Returns a dict with:
        - per_system: {system_name: {"kappa": float, "n": int, "disagreements": [...]}}
        - overall: {"kappa": float, "n": int}
    """
    # Group by system
    by_system: Dict[str, List[Tuple[int, int]]] = {}
    for r in results:
        if r.human_label is None:
            continue
        auto_val = r.metrics.get(metric)
        if auto_val is None:
            continue
        auto_bin = _binarise(auto_val)
        human_bin = int(r.human_label)
        by_system.setdefault(r.system_name, []).append((human_bin, auto_bin))

    per_system: Dict[str, Any] = {}
    all_human: List[int] = []
    all_auto: List[int] = []

    for sname, pairs in by_system.items():
        human_labels = [p[0] for p in pairs]
        auto_labels = [p[1] for p in pairs]
        kappa = cohens_kappa(human_labels, auto_labels)
        disagreements = [{"index": i, "human": h, "auto": a} for i, (h, a) in enumerate(pairs) if h != a]
        per_system[sname] = {
            "kappa": kappa,
            "n": len(pairs),
            "n_disagreements": len(disagreements),
            "disagreements": disagreements,
        }
        all_human.extend(human_labels)
        all_auto.extend(auto_labels)

    overall_kappa = cohens_kappa(all_human, all_auto) if all_human else float("nan")

    return {
        "per_system": per_system,
        "overall": {
            "kappa": overall_kappa,
            "n": len(all_human),
        },
        "metric": metric,
    }
