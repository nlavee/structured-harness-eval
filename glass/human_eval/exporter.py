"""
Human evaluation exporter.

Exports a stratified sample of evaluation results for human annotation.
Every domain receives at least 2 annotated samples; overall fraction
defaults to 0.3 (30 of 100 samples) per PLAN.md.

Output CSV columns (read-only except human_label):
  sample_id, domain, question, gold_answer, system_name,
  prediction, automated_judge_score, human_label
"""

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from glass.judges.base import EvalResult
from glass.storage.run_store import RunStore

logger = logging.getLogger(__name__)

# Human-readable columns exported for annotation
EXPORT_COLUMNS = [
    "sample_id",
    "domain",
    "question",
    "gold_answer",
    "system_name",
    "prediction",
    "automated_judge_score",
    "human_label",
]

MIN_PER_DOMAIN = 2


def export_human_eval(
    run_dir: Path,
    store: RunStore,
    results: List[EvalResult],
    fraction: float = 0.3,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export a stratified subset of results for human annotation.

    Args:
        run_dir: Path to the run directory.
        store: RunStore instance for loading raw outputs.
        results: All EvalResult objects for this run.
        fraction: Proportion of samples to export (0 < fraction ≤ 1).
        seed: Random seed for reproducible stratified sampling.
        output_path: Override output path; defaults to
                     run_dir/human_eval/export_{date}.csv.

    Returns:
        Path to the exported CSV file.
    """
    import datetime

    if not results:
        raise ValueError("No results to export for human evaluation.")

    # Use a single, deterministic primary system for annotation
    # (prefer first structured_harness, else first system alphabetically)
    system_names = sorted({r.system_name for r in results})
    primary_system = next(
        (s for s in system_names if "structured_harness" in s.lower()),
        system_names[0],
    )

    # Build lookup: {sample_id: EvalResult} for primary system
    primary_results: Dict[str, EvalResult] = {r.sample_id: r for r in results if r.system_name == primary_system}

    # Stratified sampling: group by domain, ensure MIN_PER_DOMAIN
    by_domain: Dict[str, List[str]] = defaultdict(list)
    for sid in primary_results:
        domain = primary_results[sid].domain
        by_domain[domain].append(sid)

    n_total = len(primary_results)
    n_target = max(MIN_PER_DOMAIN * len(by_domain), round(n_total * fraction))
    n_target = min(n_target, n_total)

    rng = random.Random(seed)
    selected_ids: List[str] = []

    # First guarantee MIN_PER_DOMAIN per domain
    for domain, ids in by_domain.items():
        chosen = rng.sample(ids, min(MIN_PER_DOMAIN, len(ids)))
        selected_ids.extend(chosen)

    # Then fill up to n_target from remaining
    remaining = [sid for sid in primary_results if sid not in set(selected_ids)]
    rng.shuffle(remaining)
    still_needed = max(0, n_target - len(selected_ids))
    selected_ids.extend(remaining[:still_needed])

    # Load raw outputs for prediction text
    rows = []
    for sid in sorted(selected_ids):
        result = primary_results[sid]
        try:
            raw = store.load_raw_output(primary_system, sid)
            prediction = raw.output
        except FileNotFoundError:
            prediction = ""

        rows.append(
            {
                "sample_id": sid,
                "domain": result.domain,
                "question": result.metrics.get("_question", ""),  # injected if available
                "gold_answer": result.metrics.get("_gold_answer", ""),
                "system_name": primary_system,
                "prediction": prediction,
                "automated_judge_score": result.metrics.get("judge_score", ""),
                "human_label": "",  # blank for annotators to fill
            }
        )

    # Write CSV
    if output_path is None:
        date_str = datetime.date.today().strftime("%Y%m%d")
        human_eval_dir = run_dir / "human_eval"
        human_eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = human_eval_dir / f"export_{date_str}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("[GLASS] Saved human eval export → %s (%d rows)", output_path, len(rows))
    print(f"[GLASS] Saved human eval export → {output_path} ({len(rows)} rows)")
    return output_path
