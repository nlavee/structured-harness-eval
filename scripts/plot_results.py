import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

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


def load_eval_jsons(run_dir: Path, systems: List[str]) -> List[Dict[str, Any]]:
    """Loads all per-sample evaluation JSONs for the given systems."""
    eval_dir = run_dir / "evaluation"
    eval_data = []
    
    if not eval_dir.exists():
        logger.warning(f"Evaluation directory {eval_dir} does not exist. Some judge plots will be skipped.")
        return eval_data

    for system in systems:
        sys_dir = eval_dir / system
        if not sys_dir.exists():
            continue
            
        for path in sys_dir.glob("sample_*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    eval_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                
    return eval_data


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
                ci_low = metric_data.get("ci_low", mean_val)
                ci_high = metric_data.get("ci_high", mean_val)
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

    active_metrics = []
    for m in metrics:
        m_data = df_plot[df_plot["Metric"] == m]
        if not m_data.empty and m_data["Mean"].sum() > 0:
            active_metrics.append(m)
            
    if not active_metrics:
        active_metrics = metrics
        
    df_plot = df_plot[df_plot["Metric"].isin(active_metrics)]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_plot, 
        x="Metric", 
        y="Mean", 
        hue="System",
        palette="viridis"
    )

    x_coords = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
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
    """Plots a grouped bar chart of core metrics per domain."""
    logger.info("Generating domain metrics plot...")
    
    domains_data = stats_data.get("per_domain", {})
    if not domains_data:
        return

    target_metrics = ["exact_match", "soft_recall", "judge_score", "hallucination_rate"]
    
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
        return

    df_plot = pd.DataFrame(plot_data)
    available_metrics = df_plot["Metric"].unique()
    
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 5 * len(available_metrics)), sharex=False)
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
    """Plots boxplots for numerical distributions."""
    logger.info("Generating distribution plots...")
    
    target_metrics = ["latency_s", "verbosity"]
    available_metrics = [m for m in target_metrics if m in results_df.columns and not results_df[m].isna().all()]
    
    if not available_metrics:
        return
        
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 6))
    if len(available_metrics) == 1:
        axes = [axes]
        
    for ax, metric in zip(axes, available_metrics):
        sns.boxplot(
            data=results_df,
            x="system_name",
            y=metric,
            hue="system_name",
            ax=ax,
            palette="pastel",
            legend=False
        )
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


def plot_correctness_summary(results_df: pd.DataFrame, run_dir: Path):
    """Stacked bar chart of CORRECT vs INCORRECT portions per system."""
    logger.info("Generating correctness summary plot...")
    
    if "judge_score" not in results_df.columns:
        return
        
    df_clean = results_df.dropna(subset=["judge_score"])
    if df_clean.empty:
        return
        
    # Map binary score to label
    df_clean["Correctness"] = df_clean["judge_score"].apply(lambda x: "CORRECT" if x >= 0.5 else "INCORRECT")
    
    counts = df_clean.groupby(["system_name", "Correctness"]).size().unstack(fill_value=0)
    
    # Ensure both columns exist for consistent coloring
    if "CORRECT" not in counts.columns: counts["CORRECT"] = 0
    if "INCORRECT" not in counts.columns: counts["INCORRECT"] = 0
    
    # Plot as stacked bar
    colors = {"CORRECT": "#2ecc71", "INCORRECT": "#e74c3c"}
    
    ax = counts[["CORRECT", "INCORRECT"]].plot(
        kind="bar", 
        stacked=True, 
        figsize=(10, 6),
        color=[colors["CORRECT"], colors["INCORRECT"]]
    )
    
    plt.title("Judge Correctness Breakdown by System", pad=20)
    plt.xlabel("System")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add count annotations
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
        
    plt.tight_layout()
    out_path = run_dir / "plot_correctness_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_hallucination_breakdown(eval_data: List[Dict[str, Any]], run_dir: Path):
    """Stacked bar chart per system showing proportion of SUPPORTED vs CONTRADICTED vs UNVERIFIED."""
    logger.info("Generating hallucination breakdown plot...")
    
    if not eval_data:
        return
        
    # Aggregate sentence-level labels per system
    system_counts = {}
    
    for sample in eval_data:
        sys = sample.get("system_name")
        outputs = sample.get("judge_outputs", {})
        halluc = outputs.get("hallucination", "")
        
        if not sys or not halluc:
            continue
            
        if sys not in system_counts:
            system_counts[sys] = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIED": 0}
            
        try:
            # Parse string representation of list "['SUPPORTED', ...]"
            import ast
            labels = ast.literal_eval(halluc)
            if isinstance(labels, list):
                for label in labels:
                    if label in system_counts[sys]:
                        system_counts[sys][label] += 1
        except:
            pass
            
    if not system_counts:
        return
        
    df = pd.DataFrame(system_counts).T
    
    # Calculate percentages
    df_pct = df.div(df.sum(axis=1), axis=0) * 100
    
    colors = {"SUPPORTED": "#2ecc71", "CONTRADICTED": "#e74c3c", "UNVERIFIED": "#f39c12"}
    
    ax = df_pct.plot(
        kind="barh", 
        stacked=True, 
        figsize=(12, min(4 + len(df) * 0.8, 10)),
        color=[colors["SUPPORTED"], colors["CONTRADICTED"], colors["UNVERIFIED"]]
    )
    
    plt.title("Sentence-Level Hallucination Classification (Percentage)", pad=20)
    plt.xlabel("Percentage of Total Output Sentences")
    plt.ylabel("System")
    plt.legend(title="Classification", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Only label segments > 5% to avoid clutter
    for c in ax.containers:
        labels = [f'{v.get_width():.1f}%' if v.get_width() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold')
        
    plt.tight_layout()
    out_path = run_dir / "plot_hallucination_breakdown.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_correctness_vs_hallucination(results_df: pd.DataFrame, run_dir: Path):
    """Scatter plot correlating judge_score against hallucination_rate per sample."""
    logger.info("Generating correctness vs hallucination scatter plot...")
    
    if "judge_score" not in results_df.columns or "hallucination_rate" not in results_df.columns:
        return
        
    df_clean = results_df.dropna(subset=["judge_score", "hallucination_rate"])
    if df_clean.empty:
        return
        
    plt.figure(figsize=(10, 8))
    
    # Add slight jitter to x (judge_score = 0 or 1) so points don't entirely overlap
    sns.stripplot(
        data=df_clean,
        x="judge_score",
        y="hallucination_rate",
        hue="system_name",
        jitter=0.25,
        alpha=0.6,
        size=8,
        palette="husl"
    )
    
    plt.title("Hallucination Rate vs Correctness", pad=20)
    plt.xlabel("Correctness (0 = Incorrect, 1 = Correct)")
    plt.ylabel("Hallucination Rate (Fraction of Unverified/Contradicted Sentences)")
    plt.xticks([0, 1], ["0.0\n(Incorrect)", "1.0\n(Correct)"])
    
    plt.tight_layout()
    out_path = run_dir / "plot_correctness_vs_hallucination.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_domain_judge_heatmap(results_df: pd.DataFrame, run_dir: Path):
    """Heatmap of average judge_score per (domain × system)."""
    logger.info("Generating domain judge heatmap...")
    
    if "judge_score" not in results_df.columns or "domain" not in results_df.columns:
        return
        
    pivot_df = results_df.pivot_table(
        values="judge_score", 
        index="domain", 
        columns="system_name", 
        aggfunc="mean"
    )
    
    if pivot_df.empty:
        return
        
    plt.figure(figsize=(10, min(8, max(4, len(pivot_df) * 0.8))))
    
    sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap="YlGnBu", 
        vmin=0, 
        vmax=1,
        fmt=".2f",
        cbar_kws={'label': 'Mean Correctness'}
    )
    
    plt.title("Average Correctness by Domain and System", pad=20)
    plt.ylabel("Domain")
    plt.xlabel("System")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    out_path = run_dir / "plot_domain_judge_heatmap.png"
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

    stats_path = run_dir / "statistics.json"
    results_path = run_dir / "results.csv"

    if not stats_path.exists() or not results_path.exists():
        logger.error(f"Missing required files in {run_dir}. Need both statistics.json and results.csv")
        return

    logger.info(f"Loading data from {run_dir.name}...")
    try:
        with open(stats_path, "r") as f:
            stats_data = json.load(f)
            
        results_df = pd.read_csv(results_path)
    except Exception as e:
        logger.error(f"Failed to load data files: {e}")
        return

    run_config = stats_data.get("run_config", {})
    metrics = run_config.get("metrics", [])
    systems = run_config.get("systems", [])

    if not metrics or not systems:
        logger.error("Could not find 'metrics' or 'systems' in statistics.json run_config.")
        return

    # Load detailed evaluation JSONs for deep hallucination analysis
    eval_data = load_eval_jsons(run_dir, systems)

    sns.set_theme(style="whitegrid")

    # Generate existing plots
    plot_overall_metrics(stats_data, run_dir, metrics, systems)
    plot_domain_metrics(stats_data, run_dir, systems)
    plot_distributions(results_df, run_dir, systems)
    
    # Generate new judge-focused plots
    plot_correctness_summary(results_df, run_dir)
    plot_hallucination_breakdown(eval_data, run_dir)
    plot_correctness_vs_hallucination(results_df, run_dir)
    plot_domain_judge_heatmap(results_df, run_dir)
    
    logger.info("Plot generation complete!")

if __name__ == "__main__":
    main()
