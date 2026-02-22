import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import sys

# Add current directory to path so we can import schema
sys.path.append(str(Path(__file__).parent))
from schema import AggregatedData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("Visualizer")

# Global style configurations for PhD-level output
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def enforce_ap_rh2_metadata(ax, sample_size: int):
    """AP-RH2: Always overlay the $N$ sample size explicitely to avoid hiding statistical power loss."""
    ax.annotate(f"N = {sample_size} (paired)", xy=(0.98, 0.95), xycoords='axes fraction', 
                ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=10)

def plot_forest_cis(global_stats: Dict, systems: List[str], metric: str, out_path: Path, sample_size: int):
    """
    AP-RH2: Use forest plots for primary metric CIs instead of Dynamite plots.
    """
    logger.info(f"Generating Forest Plot for {metric}")
    fig, ax = plt.subplots(figsize=(8, 4))
    
    y_positions = np.arange(len(systems))
    means = []
    errors_low_list = []
    errors_high_list = []
    
    for system in systems:
        # AP-RH5 extracts data natively. The aggregator maps the full payload.
        # Systems are mapped by the full `run_id` directly to track independent stats.
        stats = global_stats.get(system, {}).get(metric, {})
        if not stats:
            means.append(np.nan)
            errors_low_list.append(np.nan)
            errors_high_list.append(np.nan)
            continue
            
        means.append(stats['mean'])
        # matplotlib errorbar wants relative distance from mean, not absolute values
        errors_low = stats['mean'] - stats.get('ci_low', stats['mean'])
        errors_high = stats.get('ci_high', stats['mean']) - stats['mean']
        
        errors_low_list.append(errors_low)
        errors_high_list.append(errors_high)

    # Convert to 2xN array for matplotlib
    errors = np.array([errors_low_list, errors_high_list])
    
    ax.errorbar(means, y_positions, xerr=errors, fmt='o', color='black', capsize=5, capthick=2, markersize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(systems)
    ax.set_xlabel(f"{metric} (95% CI)")
    ax.set_title(f"Bootstrap Confidence Intervals: {metric}")
    
    enforce_ap_rh2_metadata(ax, sample_size)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_paired_violin(df: pd.DataFrame, systems: List[str], metrics: List[str], out_path: Path):
    """
    AP-RH2: Use Violin + Swarm plots for continuous distributions rather than bar charts.
    """
    logger.info(f"Generating Violin distributions for {metrics}")
    
    # Reshape the dataframe for seaborn (melt)
    melted_data = []
    for _, row in df.iterrows():
        sample_id = row['sample_id']
        for system in systems:
            for metric in metrics:
                # The data aggregator renames core df columns to `{system}_{metric}`
                col_name = f"{system}_{metric}"
                if col_name in row and pd.notna(row[col_name]):
                    try:
                        val = float(row[col_name])
                        melted_data.append({
                            "sample_id": sample_id,
                            "System": system,
                            "Metric": metric,
                            "Score": val
                        })
                    except (ValueError, TypeError):
                        continue
                    
    melted_df = pd.DataFrame(melted_data)
    
    if melted_df.empty:
        logger.warning(f"No numeric data found for violin distributions of {metrics}")
        return

    g = sns.catplot(data=melted_df, x="System", y="Score", col="Metric", 
                    hue="System", legend=False,
                    kind="violin", inner=None, palette="muted", 
                    sharey=False, height=5, aspect=0.8)
    
    # Overlay swarm plot to show raw points
    g.map_dataframe(sns.stripplot, x="System", y="Score", color="black", alpha=0.5, size=3)
    
    # Add titles and N
    for i, ax in enumerate(g.axes.flat):
        ax.set_title(f"Distribution: {metrics[i]}")
        enforce_ap_rh2_metadata(ax, len(df))
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_behavior_radar(global_stats: Dict, systems: List[str], out_path: Path, sample_size: int):
    """
    Generates a radar chart comparing scalar behavioral metrics between systems.
    Behavior metrics (e.g., verbosity) have wildly different scales, so we normalize.
    """
    logger.info("Generating Behavioral Radar Chart")
    
    behavior_metrics = ["soft_recall", "latency_s", "verbosity"]
    
    # Build data matrix
    data = []
    max_vals = []
    for metric in behavior_metrics:
        vals = []
        for sys in systems:
            vals.append(global_stats.get(sys, {}).get(metric, {}).get("mean", 0.0))
            
        max_val = max(vals) if vals and max(vals) > 0 else 1.0
        max_vals.append(max_val)
        # Normalize to 0-1 for radar overlay
        norm_vals = [v / max_val for v in vals]
        data.append(norm_vals)
        
    data = np.array(data) # shape: (len(behavior_metrics), len(systems))
    
    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, len(behavior_metrics), endpoint=False).tolist()
    angles += angles[:1] # close the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for i, system in enumerate(systems):
        values = data[:, i].tolist()
        values += values[:1] # close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=system)
        ax.fill(angles, values, alpha=0.25)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(behavior_metrics)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([]) # Hide radial ticks
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Normalized Behavioral Profile")
    enforce_ap_rh2_metadata(ax, sample_size)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_domain_heatmap(domain_stats: Dict, systems: List[str], metric: str, out_path: Path):
    """
    Plots a heatmap for a specific metric across different domains and systems.
    """
    logger.info(f"Generating Domain Heatmap for {metric}")
    
    # Collect all unique domains
    domains = set()
    for sys_stats in domain_stats.values():
        domains.update(sys_stats.keys())
    domains = sorted(list(domains))
    
    if not domains:
        logger.warning(f"No domains found for heatmap {metric}")
        return

    data = []
    for domain in domains:
        row = []
        for sys in systems:
            val = domain_stats.get(sys, {}).get(domain, {}).get(metric, {}).get("mean", np.nan)
            row.append(val)
        data.append(row)
        
    df = pd.DataFrame(data, index=domains, columns=systems)
    
    # Drop rows that are entirely NaN
    df = df.dropna(how='all')
    if df.empty:
        logger.warning(f"No valid data found for heatmap {metric}")
        return
        
    plt.figure(figsize=(10, len(df) * 0.8 + 2))
    cmap = "Blues" if metric in ["judge_score", "exact_match", "soft_recall"] else "Reds"
    
    sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", cbar_kws={'label': f'{metric} (Mean)'})
    plt.title(f"Domain Breakdown: {metric}")
    plt.ylabel("Domain")
    plt.xlabel("System / Run")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_paired_difference(df: pd.DataFrame, systems: List[str], metrics: List[str], out_path: Path):
    """
    AP-RH2: Paired Sample Difference (Waterfall / Scatter).
    Calculates the delta (System A - System B) per sample and plots the distribution of deltas.
    Useful for seeing if a system fixes failures or just fails on different samples.
    """
    logger.info(f"Generating Paired Difference Plot for {metrics} between '{systems[0]}' and '{systems[1]}'")
    
    if len(systems) < 2:
        logger.warning("Need at least 2 systems for paired difference plot.")
        return
        
    # We compare the first two systems in the list (typically structured harness vs baseline)
    sys_a = systems[0]
    sys_b = systems[1]
    
    # Restructure data for sns distribution plots
    delta_data = []
    
    for metric in metrics:
        col_a = f"{sys_a}_{metric}"
        col_b = f"{sys_b}_{metric}"
        
        if col_a in df.columns and col_b in df.columns:
            # Dropna for the pair
            valid_pairs = df[[col_a, col_b, 'sample_id']].dropna()
            for _, row in valid_pairs.iterrows():
                try:
                    val_a = float(row[col_a])
                    val_b = float(row[col_b])
                    delta = val_a - val_b
                    
                    delta_data.append({
                        "Metric": metric,
                        "Delta": delta,
                        "sample_id": row['sample_id']
                    })
                except (ValueError, TypeError):
                    continue
                    
    delta_df = pd.DataFrame(delta_data)
    if delta_df.empty:
        logger.warning("No valid numeric pairs found for difference plotting.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # We use a stripplot combined with a boxplot to show the distribution of deltas
    sns.boxplot(data=delta_df, x="Metric", y="Delta", color="lightgray", showfliers=False)
    sns.stripplot(data=delta_df, x="Metric", y="Delta", alpha=0.6, jitter=True)
    
    # Add a zero line
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label="No Difference")
    
    plt.title(f"Paired Difference: {sys_a} (A) - {sys_b} (B)")
    plt.ylabel(r"$\Delta$ (A - B)")
    plt.xlabel("Metric")
    plt.legend()
    
    enforce_ap_rh2_metadata(plt.gca(), len(delta_df) // len(metrics) if len(metrics) > 0 else 0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Research Harness Visualizer enforcing AP-RH2.")
    parser.add_argument("--data", required=True, help="Path to aggregated_data.json")
    parser.add_argument("--out-dir", default="research_insights/figures", help="Directory to save figures.")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.data, "r") as f:
        raw_json = f.read()
        
    try:
        validated_payload = AggregatedData.model_validate_json(raw_json)
        payload = validated_payload.model_dump()
    except Exception as e:
        logger.error(f"FATAL: The incoming aggregated data failed Pydantic schema validation: {e}")
        sys.exit(1)
        
    systems = payload["metadata"]["systems"]
    sample_size = payload["metadata"]["paired_sample_n"]
    global_stats = payload["global_statistics"]
    domain_stats = payload.get("domain_statistics", {})
    df = pd.DataFrame(payload["joined_dataframe"])
    
    # Forest Plots for Primary Metrics
    plot_forest_cis(global_stats, systems, "judge_score", out_dir / "forest_judge_score.png", sample_size)
    plot_forest_cis(global_stats, systems, "hallucination_rate", out_dir / "forest_hallucination_rate.png", sample_size)
    
    # Continuous metrics
    continuous_metrics = ["hallucination_rate", "soft_recall", "latency_s"]
    plot_paired_violin(df, systems, continuous_metrics, out_dir / "violin_distributions.png")
    
    # AP-RH2 Paired Differences
    if len(systems) >= 2:
        diff_metrics = ["judge_score", "hallucination_rate", "soft_recall"]
        plot_paired_difference(df, systems[:2], diff_metrics, out_dir / "paired_differences.png")
    
    # Behavioral Radar Chart
    plot_behavior_radar(global_stats, systems, out_dir / "radar_behavior.png", sample_size)
    
    # Domain Breakdowns
    if domain_stats:
        metrics_to_plot = ["judge_score", "exact_match", "hallucination_rate"]
        for metric in metrics_to_plot:
            plot_domain_heatmap(domain_stats, systems, metric, out_dir / f"domain_heatmap_{metric}.png")
            
    logger.info(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()
