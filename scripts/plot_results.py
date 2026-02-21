import argparse
import json
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Configure logging
logging.basicConfig(level=logging.INFO, format="[GLASS Plotter] %(message)s")
logger = logging.getLogger(__name__)


def get_latest_run_dir(runs_dir: Path) -> Path:
    """Finds the most recently created run directory containing statistics.json."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    valid_dirs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and (d / "statistics.json").exists():
            valid_dirs.append(d)

    if not valid_dirs:
        raise FileNotFoundError(f"No run directories with statistics.json found in {runs_dir}")

    # Return the directory with the most recent modification time
    return max(valid_dirs, key=lambda p: p.stat().st_mtime)


def plot_overall_metrics(stats_data: dict, run_dir: Path, metrics: List[str], systems: List[str]):
    """Plots a bar chart of average metrics for all systems."""
    logger.info("Generating overall metrics plot...")
    
    # Extract data for plotting
    plot_data = []
    for system in systems:
        sys_stats = stats_data.get("system_stats", {}).get(system, {})
        for metric in metrics:
            metric_data = sys_stats.get(metric, {})
            # Only include metrics that have a valid mean
            mean_val = metric_data.get("mean")
            if mean_val is not None:
                # Calculate error margin (difference between mean and ci bounds)
                # If CI bounds aren't available, default to 0
                ci_low = metric_data.get("ci_low", mean_val)
                ci_high = metric_data.get("ci_high", mean_val)
                # Handle null CIs gracefully
                if ci_low is None: ci_low = mean_val
                if ci_high is None: ci_high = mean_val
                
                err_low = mean_val - ci_low
                err_high = ci_high - mean_val

                plot_data.append({
                    "System": system,
                    "Metric": metric,
                    "Mean": mean_val,
                    "Err_Low": err_low,
                    "Err_High": err_high
                })

    if not plot_data:
        logger.warning("No valid metric data found for overall plot. Skipping.")
        return

    df_plot = pd.DataFrame(plot_data)

    # Filter out metrics where all systems have 0 or None to declutter the chart
    active_metrics = []
    for m in metrics:
        m_data = df_plot[df_plot["Metric"] == m]
        if not m_data.empty and m_data["Mean"].sum() > 0:
            active_metrics.append(m)
            
    if not active_metrics:
        active_metrics = metrics # fallback if all are 0
        
    df_plot = df_plot[df_plot["Metric"].isin(active_metrics)]

    # Set up the plot
    plt.figure(figsize=(12, 6))
    
    # We use seaborn barplot for grouped bars
    ax = sns.barplot(
        data=df_plot, 
        x="Metric", 
        y="Mean", 
        hue="System",
        palette="viridis"
    )

    # Add error bars manually since we have asymmetric CIs from bootstrap
    x_coords = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    
    # Seaborn plots hue categories sequentially. We need to match coordinates to dataframe rows.
    # The order of patches matches the order of hues (systems), then x-categories (metrics)
    sorted_df = df_plot.sort_values(by=["System", "Metric"])
    
    if len(x_coords) == len(sorted_df):
        plt.errorbar(
            x=x_coords,
            y=y_coords,
            yerr=[sorted_df["Err_Low"], sorted_df["Err_High"]],
            fmt="none",
            c="black",
            capsize=4
        )

    plt.title("Overall Metrics by System (with 95% Bootstrap CI)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_path = run_dir / "plot_overall_metrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_domain_metrics(stats_data: dict, run_dir: Path, systems: List[str]):
    """Plots a grouped bar chart of exact_match and soft_recall per domain."""
    logger.info("Generating domain metrics plot...")
    
    domains_data = stats_data.get("per_domain", {})
    if not domains_data:
        logger.info("No per_domain data found. Skipping domain plots.")
        return

    # Let's focus on accuracy metrics for the domain breakdown
    target_metrics = ["exact_match", "soft_recall"]
    
    plot_data = []
    for domain, sys_dict in domains_data.items():
        for system in systems:
            metrics_dict = sys_dict.get(system, {})
            for metric in target_metrics:
                mean_val = metrics_dict.get(metric, {}).get("mean")
                if mean_val is not None:
                    plot_data.append({
                        "Domain": domain,
                        "System": system,
                        "Metric": metric,
                        "Mean": mean_val
                    })

    if not plot_data:
        logger.warning("No valid exact_match/soft_recall data in domains. Skipping.")
        return

    df_plot = pd.DataFrame(plot_data)

    # Create a subplot for each target metric
    available_metrics = df_plot["Metric"].unique()
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 5 * len(available_metrics)), sharex=False)
    
    # Handle single subplot case
    if len(available_metrics) == 1:
        axes = [axes]
        
    for ax, metric in zip(axes, available_metrics):
        metric_df = df_plot[df_plot["Metric"] == metric]
        sns.barplot(
            data=metric_df,
            x="Domain",
            y="Mean",
            hue="System",
            ax=ax,
            palette="Set2"
        )
        ax.set_title(f"{metric} by Domain")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out_path = run_dir / "plot_domain_metrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_distributions(results_df: pd.DataFrame, run_dir: Path, systems: List[str]):
    """Plots boxplots for numerical distributions like latency_s and verbosity."""
    logger.info("Generating distribution plots...")
    
    # Identify numerical metrics that are continuous
    target_metrics = ["latency_s", "verbosity"]
    
    # Filter only metrics that actually exist in the CSV and have data
    available_metrics = [m for m in target_metrics if m in results_df.columns and not results_df[m].isna().all()]
    
    if not available_metrics:
        logger.warning("No valid distribution metrics found in results.csv. Skipping.")
        return
        
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 6))
    
    # Handle single subplot case
    if len(available_metrics) == 1:
        axes = [axes]
        
    for ax, metric in zip(axes, available_metrics):
        sns.boxplot(
            data=results_df,
            x="system_name",
            y=metric,
            ax=ax,
            palette="pastel"
        )
        # Add stripplot to show individual data points
        sns.stripplot(
            data=results_df,
            x="system_name",
            y=metric,
            ax=ax,
            color=".3",
            alpha=0.5,
            jitter=True
        )
        ax.set_title(f"Distribution of {metric}")
        ax.set_xlabel("System")
        ax.set_ylabel(metric)

    plt.tight_layout()
    out_path = run_dir / "plot_distributions.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from GLASS evaluation run outputs.")
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to the specific run directory. If omitted, uses the latest valid run in 'runs/'.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Determine run directory
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        runs_base = project_root / "runs"
        try:
            run_dir = get_latest_run_dir(runs_base)
            logger.info(f"Auto-selected latest run: {run_dir.name}")
        except FileNotFoundError as e:
            logger.error(e)
            return

    # Validate required files
    stats_path = run_dir / "statistics.json"
    results_path = run_dir / "results.csv"

    if not stats_path.exists() or not results_path.exists():
        logger.error(f"Missing required files in {run_dir}. Need both statistics.json and results.csv")
        return

    # Load data
    logger.info(f"Loading data from {run_dir.name}...")
    try:
        with open(stats_path, "r") as f:
            stats_data = json.load(f)
            
        results_df = pd.read_csv(results_path)
    except Exception as e:
        logger.error(f"Failed to load data files: {e}")
        return

    # Extract dynamic configuration
    run_config = stats_data.get("run_config", {})
    metrics = run_config.get("metrics", [])
    systems = run_config.get("systems", [])

    if not metrics or not systems:
        logger.error("Could not find 'metrics' or 'systems' in statistics.json run_config.")
        return

    # Set visualization style
    sns.set_theme(style="whitegrid")

    # Generate plots
    plot_overall_metrics(stats_data, run_dir, metrics, systems)
    plot_domain_metrics(stats_data, run_dir, systems)
    plot_distributions(results_df, run_dir, systems)
    
    logger.info("Plot generation complete!")

if __name__ == "__main__":
    main()
