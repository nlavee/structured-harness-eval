import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from glass.judges.base import EvalResult
from glass.statistics.bootstrap import compute_ci


def generate_summary(
    results: List[EvalResult],
    output_path: Path,
    stats_path: Optional[Path] = None,
    seed: int = 42,
) -> None:
    if not results:
        with open(output_path, "w") as f:
            f.write("# GLASS Evaluation Summary\n\nNo results found.")
        return

    df = pd.DataFrame([r.model_dump() for r in results])

    # Flatten metrics dict column
    metrics_df = pd.json_normalize(df["metrics"])
    df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)

    metric_cols = [
        c
        for c in df.columns
        if c not in ["sample_id", "system_name", "domain", "judge_model", "judge_outputs", "human_label"]
    ]

    md = ["# GLASS Evaluation Summary\n"]

    # --- Aggregate Table ---
    md.append("## Aggregate Metrics\n")
    md.append(
        "> 95% bootstrap CI shown as [lower, upper]. See `statistics.json` for significance tests and effect sizes.\n"
    )
    agg_rows = []
    for system in df["system_name"].unique():
        sys_df = df[df["system_name"] == system]
        row = {"System": system}
        for metric in metric_cols:
            vals = sys_df[metric].dropna().tolist()
            if not vals:
                row[metric] = "N/A"
                continue
            mean = sum(vals) / len(vals)
            ci_low, ci_high = compute_ci(vals, seed=seed)
            row[metric] = f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    md.append(agg_df.to_markdown(index=False))
    md.append("\n")

    # --- Statistical Highlights (from statistics.json if present) ---
    if stats_path and Path(stats_path).exists():
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            primary = stats.get("primary_hypothesis", {})
            if primary:
                md.append("## Primary Hypothesis Results\n")
                for comparison, res in primary.items():
                    p = res.get("p_value", "N/A")
                    r = res.get("effect_size", "N/A")
                    sig = "✓ significant" if isinstance(p, float) and p < 0.05 else "✗ not significant"
                    md.append(
                        f"- **{comparison}**: p={p:.4f}, r={r:.3f} ({sig})\n"
                        if isinstance(p, float)
                        else f"- **{comparison}**: {p}\n"
                    )
        except Exception:
            pass  # Stats file malformed or missing; skip highlights

    # --- Per-Domain Breakdown ---
    md.append("## Per-Domain Breakdown (Exploratory)\n")
    md.append(
        "> **Note (AP-19):** Per-domain samples are typically N≈14, which is insufficient "
        "for statistical significance claims. These numbers are descriptive only.\n"
    )
    for domain in sorted(df["domain"].unique()):
        dom_df = df[df["domain"] == domain]
        n_samples = dom_df["sample_id"].nunique()
        md.append(f"### Domain: {domain} (N={n_samples} samples)\n")
        dom_rows = []
        for system in df["system_name"].unique():
            sys_df = dom_df[dom_df["system_name"] == system]
            row = {"System": system}
            for metric in metric_cols:
                vals = sys_df[metric].dropna().tolist()
                row[metric] = f"{sum(vals) / len(vals):.3f}" if vals else "N/A"
            dom_rows.append(row)
        md.append(pd.DataFrame(dom_rows).to_markdown(index=False))
        md.append("\n")

    with open(output_path, "w") as f:
        f.write("\n".join(md))
