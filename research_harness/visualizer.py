import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import sys
from rich.logging import RichHandler

# Add current directory to path so we can import schema
sys.path.append(str(Path(__file__).parent))
from schema import AggregatedData
from naming import PlotType, get_plot_filename

# Configure logging to write to STDOUT so the orchestrator can capture it
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=None, show_time=False, show_path=False, markup=True)]
)
logger = logging.getLogger("Visualizer")

# Global style configurations for PhD-level output
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def enforce_ap_rh2_metadata(ax, sample_size: int):
    """AP-RH2: Always overlay the $N$ sample size explicitely to avoid hiding statistical power loss."""
    ax.annotate(f"N = {sample_size} (paired)", xy=(0.98, 0.95), xycoords='axes fraction', 
                ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=10)

def shorten_label(name: str, max_len: int = 25, aliases: Optional[Dict[str, str]] = None) -> str:
    """Shortens long run IDs or uses an alias if available."""
    if aliases and name in aliases:
        return aliases[name]
    
    if len(name) <= max_len:
        return name
    
    # Common run id format: YYYYMMDD_HHMMSS_name
    # We want to keep the date bit and the name bit
    parts = name.split('_')
    if len(parts) >= 3:
        # Keep YYYYMMDD and the last part
        short_name = f"{parts[0]}...{parts[-1]}"
        if len(short_name) <= max_len:
            return short_name
            
    return name[:max_len-3] + "..."

def plot_forest_cis(global_stats: Dict, systems: List[str], metric: str, out_path: Path, sample_size: int, aliases: Optional[Dict[str, str]] = None):
    """
    AP-RH2: Use forest plots for primary metric CIs instead of Dynamite plots.
    """
    logger.info(f"Generating Forest Plot for {metric}")
    fig, ax = plt.subplots(figsize=(10, max(4, len(systems) * 0.8)))
    
    y_positions = np.arange(len(systems))
    means = []
    errors_low_list = []
    errors_high_list = []
    
    for system in systems:
        stats = global_stats.get(system, {}).get(metric, {})
        if not stats:
            means.append(np.nan)
            errors_low_list.append(np.nan)
            errors_high_list.append(np.nan)
            continue
            
        means.append(stats['mean'])
        errors_low = stats['mean'] - stats.get('ci_low', stats['mean'])
        errors_high = stats.get('ci_high', stats['mean']) - stats['mean']
        
        errors_low_list.append(errors_low)
        errors_high_list.append(errors_high)

    errors = np.array([errors_low_list, errors_high_list])
    
    ax.errorbar(means, y_positions, xerr=errors, fmt='o', color='black', capsize=5, capthick=2, markersize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([shorten_label(s, aliases=aliases) for s in systems])
    ax.set_xlabel(f"{metric} (95% CI)")
    ax.set_title(f"Bootstrap Confidence Intervals: {metric}")
    
    enforce_ap_rh2_metadata(ax, sample_size)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_paired_violin(df: pd.DataFrame, systems: List[str], metrics: List[str], out_path: Path, aliases: Optional[Dict[str, str]] = None):
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
                col_name = f"{system}_{metric}"
                if col_name in row and pd.notna(row[col_name]):
                    try:
                        val = float(row[col_name])
                        melted_data.append({
                            "sample_id": sample_id,
                            "System": shorten_label(system, max_len=20, aliases=aliases),
                            "Metric": metric,
                            "Score": val
                        })
                    except (ValueError, TypeError):
                        continue
                    
    melted_df = pd.DataFrame(melted_data)
    
    if melted_df.empty:
        logger.warning(f"No numeric data found for violin distributions of {metrics}")
        return

    # Use plt.subplots instead of sns.catplot for better layout control and to avoid warnings
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6), squeeze=False)
    
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        metric_df = melted_df[melted_df["Metric"] == metric]
        
        if metric_df.empty:
            continue
            
        sns.violinplot(data=metric_df, x="System", y="Score", hue="System", 
                       inner=None, palette="muted", ax=ax, legend=False)
        sns.stripplot(data=metric_df, x="System", y="Score", color="black", 
                      alpha=0.5, size=3, ax=ax)
        
        ax.set_title(f"Distribution: {metric}")
        enforce_ap_rh2_metadata(ax, len(df))
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_behavior_radar(global_stats: Dict, systems: List[str], out_path: Path, sample_size: int, aliases: Optional[Dict[str, str]] = None):
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
    
    # Increase figure size for long vertical legend
    fig, ax = plt.subplots(figsize=(8, 10), subplot_kw=dict(polar=True))
    
    for i, system in enumerate(systems):
        values = data[:, i].tolist()
        values += values[:1] # close the loop
        # Use full alias/name in Radar chart legend per user request
        label = shorten_label(system, max_len=100, aliases=aliases)
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=label)
        ax.fill(angles, values, alpha=0.25)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(behavior_metrics)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([]) # Hide radial ticks
    
    # Move legend below the chart and make it vertical (ncol=1)
    plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), ncol=1, frameon=True)
    plt.title("Normalized Behavioral Profile", pad=30)
    enforce_ap_rh2_metadata(ax, sample_size)
    
    # Adjust tight_layout to handle the legend below
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_domain_heatmap(domain_stats: Dict, systems: List[str], metric: str, out_path: Path, aliases: Optional[Dict[str, str]] = None):
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
        
    short_systems = [shorten_label(s, aliases=aliases) for s in systems]
    df = pd.DataFrame(data, index=domains, columns=short_systems)
    
    # Drop rows that are entirely NaN
    df = df.dropna(how='all')
    if df.empty:
        logger.warning(f"No valid data found for heatmap {metric}")
        return
        
    plt.figure(figsize=(max(8, len(systems) * 1.5), len(df) * 0.8 + 2))
    cmap = "Blues" if metric in ["judge_score", "exact_match", "soft_recall"] else "Reds"
    
    sns.heatmap(df, annot=True, cmap=cmap, fmt=".2f", cbar_kws={'label': f'{metric} (Mean)'})
    plt.title(f"Domain Breakdown: {metric}")
    plt.ylabel("Domain")
    plt.xlabel("System / Run")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_paired_difference(df: pd.DataFrame, systems: List[str], metrics: List[str], out_path: Path, aliases: Optional[Dict[str, str]] = None):
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
    
    plt.title(f"Paired Difference: {shorten_label(sys_a, aliases=aliases)} (A) - {shorten_label(sys_b, aliases=aliases)} (B)")
    plt.ylabel(r"$\Delta$ (A - B)")
    plt.xlabel("Metric")
    plt.legend(title="Reference", labels=["No Difference", "Samples"])
    
    enforce_ap_rh2_metadata(plt.gca(), len(delta_df) // len(metrics) if len(metrics) > 0 else 0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_win_rate_matrix(win_rates: Dict, systems: List[str], metric: str, out_path: Path, aliases: Optional[Dict[str, str]] = None):
    """
    Plots a heatmap matrix of pairwise win rates (Row > Col).
    """
    logger.info(f"Generating Win Rate Matrix for {metric}")
    
    if not win_rates:
        logger.warning(f"No win rate data for {metric}")
        return
        
    data = []
    for row_sys in systems:
        row = []
        for col_sys in systems:
            val = win_rates.get(row_sys, {}).get(col_sys, 0.0)
            row.append(val)
        data.append(row)
        
    short_systems = [shorten_label(s, aliases=aliases) for s in systems]
    df = pd.DataFrame(data, index=short_systems, columns=short_systems)
    
    plt.figure(figsize=(max(8, len(systems) * 1.5), max(6, len(systems) * 1.0)))
    sns.heatmap(df, annot=True, cmap="Greens", fmt=".2f", vmin=0, vmax=1,
                cbar_kws={'label': f'Win Rate (Row > Col) on {metric}'})
    
    plt.title(f"Pairwise Win Rates: {metric}")
    plt.ylabel("System A (Winner)")
    plt.xlabel("System B (Loser)")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_significance_heatmap(p_values: Dict, systems: List[str], metric: str, out_path: Path, aliases: Optional[Dict[str, str]] = None):
    """
    Plots a heatmap of Holm-Bonferroni corrected p-values.
    """
    logger.info(f"Generating Significance Heatmap for {metric}")
    
    if not p_values:
        logger.warning(f"No p-value data for {metric}")
        return
        
    # Initialize with 1.0 (not significant)
    data = np.ones((len(systems), len(systems)))
    
    # Fill upper triangle
    for i, sys_a in enumerate(systems):
        for j, sys_b in enumerate(systems):
            if i >= j: continue
            
            pair_key = f"{sys_a} vs {sys_b}"
            # Try both orders just in case
            p_val = p_values.get(pair_key)
            if p_val is None:
                p_val = p_values.get(f"{sys_b} vs {sys_a}", 1.0)
                
            data[i, j] = p_val
            data[j, i] = p_val # Symmetric
            
    short_systems = [shorten_label(s, aliases=aliases) for s in systems]
    df = pd.DataFrame(data, index=short_systems, columns=short_systems)
    
    plt.figure(figsize=(max(8, len(systems) * 1.5), max(6, len(systems) * 1.0)))
    
    # Custom cmap: Green for significant (<0.05), Grey for non-significant
    # We use a masked array or just a sequential map where low is significant
    sns.heatmap(df, annot=True, cmap="Reds_r", fmt=".3f", vmin=0, vmax=0.1,
                cbar_kws={'label': 'Corrected p-value (Holm-Bonferroni)'})
    
    plt.title(f"Statistical Significance: {metric}")
    plt.xticks(rotation=45, ha='right')
    
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
        
    aliases = payload["metadata"].get("aliases", {})
    systems = payload["metadata"]["systems"]
    sample_size = payload["metadata"]["paired_sample_n"]
    global_stats = payload["global_statistics"]
    domain_stats = payload.get("domain_statistics", {})
    df = pd.DataFrame(payload["joined_dataframe"])
    
    # Dynamically extract all metrics present in the win_rate_matrix (computed by compare_runs)
    win_rates = payload.get("win_rate_matrix", {})
    p_values = payload.get("pairwise_significance", {})
    
    intersection_metrics = list(win_rates.keys())
    if not intersection_metrics:
        logger.warning("No dynamic metrics found in the payload win_rate_matrix. Visualizations may be empty.")
    
    # 1. Forest Plots (Primary CIs) for every metric
    for metric in intersection_metrics:
        out_path = out_dir / get_plot_filename(PlotType.FOREST, metric)
        plot_forest_cis(global_stats, systems, metric, out_path, sample_size, aliases=aliases)
    
    # 2. Continuous distributions (Violin Plots) - Plotted individually to prevent clustering
    for metric in intersection_metrics:
        out_path = out_dir / get_plot_filename(PlotType.VIOLIN, metric)
        plot_paired_violin(df, systems, [metric], out_path, aliases=aliases)
    
    # 3. Paired Differences - Plotted individually
    if len(systems) >= 2:
        for metric in intersection_metrics:
            out_path = out_dir / get_plot_filename(PlotType.PAIRED_DIFF, metric)
            plot_paired_difference(df, systems[:2], [metric], out_path, aliases=aliases)
    
    # 4. Win Rates & Significance Heatmaps
    for metric in intersection_metrics:
        if metric in win_rates:
            out_path = out_dir / get_plot_filename(PlotType.WIN_RATE, metric)
            plot_win_rate_matrix(win_rates.get(metric), systems, metric, out_path, aliases=aliases)
        if metric in p_values:
            out_path = out_dir / get_plot_filename(PlotType.SIGNIFICANCE, metric)
            plot_significance_heatmap(p_values.get(metric), systems, metric, out_path, aliases=aliases)
 
    # 5. Behavioral Radar Chart
    radar_path = out_dir / get_plot_filename(PlotType.RADAR)
    plot_behavior_radar(global_stats, systems, radar_path, sample_size, aliases=aliases)
    
    # 6. Domain Breakdowns (For all dynamic metrics)
    if domain_stats:
        for metric in intersection_metrics:
            out_path = out_dir / get_plot_filename(PlotType.DOMAIN_HEATMAP, metric)
            plot_domain_heatmap(domain_stats, systems, metric, out_path, aliases=aliases)
            
    logger.info(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()
