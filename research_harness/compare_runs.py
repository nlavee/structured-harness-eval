import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import sys
from rich.logging import RichHandler

# Add current directory to path so we can import schema
sys.path.append(str(Path(__file__).parent))
from schema import AggregatedData

# Configure logging to write to STDOUT so the orchestrator can capture it
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=None, show_time=False, show_path=False, markup=True)]
)
logger = logging.getLogger("CompareRuns")

class RunData:
    def __init__(self, run_id: str, path: Path):
        self.run_id = run_id
        self.path = path
        
        # AP-RH5: Don't compute global stats ourselves. Extract canonically from statistics.json
        with open(path / "statistics.json", "r") as f:
            self.stats = json.load(f)
            
        self.results_df = pd.read_csv(path / "results.csv")
        
        with open(path / "config.yaml", "r") as f:
            self.config = f.read()
            
    def get_system_name(self) -> str:
        """Returns the primary system tested in this run."""
        return self.stats["run_config"]["systems"][0]

def load_runs(run_ids: List[str], base_dir: str = "runs") -> List[RunData]:
    runs = []
    base_path = Path(base_dir)
    for run_id in run_ids:
        run_path = base_path / run_id
        if not run_path.exists():
            logger.error(f"Run directory not found: {run_path}")
            raise FileNotFoundError(f"Run {run_id} not found.")
        runs.append(RunData(run_id, run_path))
    return runs

def enforce_ap_rh1(runs_data: List[RunData]) -> pd.DataFrame:
    """
    AP-RH1: Ignoring Sample Alignment (Unpaired Comparisons).
    Performs a strict inner join on sample_id across all runs.
    Ensures that paired metrics only operate on the exact intersection.
    """
    logger.info("Enforcing AP-RH1: Computing strict inner join on sample_id across all runs.")
    
    # Compute the strict intersection of all metric columns across all runs
    metric_cols_list = [set(r.results_df.columns) - {"sample_id", "domain", "system_name"} for r in runs_data]
    intersection_metrics = list(set.intersection(*metric_cols_list))
    
    # Base configuration starts with just the sample_id and domain
    base_df = runs_data[0].results_df[["sample_id", "domain"]].copy()
    joined_df = base_df
    
    for run in runs_data:
        df_run = run.results_df.copy()
        
        # We use the run_id as the unique prefix per user request for clarity in visuals
        prefix = run.run_id
        
        rename_mapping = {col: f"{prefix}_{col}" for col in intersection_metrics}
        
        # Only select the intersection columns to avoid KeyErrors on older runs
        df_run_metrics = df_run[["sample_id", "domain"] + intersection_metrics].rename(columns=rename_mapping)
        
        joined_df = pd.merge(joined_df, df_run_metrics, on=["sample_id", "domain"], how="inner")
        
    logger.info(f"Intersection complete. {len(joined_df)} samples remained across all runs (down from {len(runs_data[0].results_df)}).")
    return joined_df

def extract_global_stats(runs_data: List[RunData]) -> Dict[str, Dict[str, dict]]:
    """
    AP-RH5: Re-computing Core Statistics Differently.
    Extract the canonical global stats from statistics.json directly.
    Returns: Dict[run_id][metric_name] = stats_dict
    """
    extracted_stats = {}
    for run in runs_data:
        system_name = run.get_system_name()
        # Ensure we only fetch global system_stats, not per-domain ones to respect power rules
        extracted_stats[run.run_id] = run.stats.get("system_stats", {}).get(system_name, {})
    return extracted_stats

def extract_domain_stats(runs_data: List[RunData]) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """
    Extracts per_domain statistics from statistics.json directly.
    Returns: Dict[run_id][domain_name][metric_name] = stats_dict
    """
    extracted_domain_stats = {}
    for run in runs_data:
        system_name = run.get_system_name()
        run_domain_stats = {}
        for domain, d_stats in run.stats.get("per_domain", {}).items():
            run_domain_stats[domain] = d_stats.get(system_name, {})
        extracted_domain_stats[run.run_id] = run_domain_stats
    return extracted_domain_stats

def compute_win_rates(df: pd.DataFrame, systems: List[str], metric: str = "judge_score") -> Dict[str, Dict[str, float]]:
    """
    Computes pairwise win rates (Row > Column) for a given metric.
    Returns: matrix[row_sys][col_sys] = win_rate (float)
    """
    matrix = {}
    for row_sys in systems:
        matrix[row_sys] = {}
        for col_sys in systems:
            if row_sys == col_sys:
                matrix[row_sys][col_sys] = 0.0
                continue
                
            col_a = f"{row_sys}_{metric}"
            col_b = f"{col_sys}_{metric}"
            
            if col_a not in df.columns or col_b not in df.columns:
                matrix[row_sys][col_sys] = 0.0
                continue
                
            # Drop NaNs
            valid_df = df[[col_a, col_b]].dropna()
            if valid_df.empty:
                matrix[row_sys][col_sys] = 0.0
                continue
                
            wins = (valid_df[col_a] > valid_df[col_b]).sum()
            total = len(valid_df)
            matrix[row_sys][col_sys] = float(wins / total) if total > 0 else 0.0
            
    return matrix

def compute_pairwise_significance(df: pd.DataFrame, systems: List[str], metric: str = "judge_score") -> Dict[str, float]:
    """
    Computes pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
    Returns: Dict["SysA vs SysB"] = corrected_p_value
    """
    p_values = {}
    comparisons = []
    
    # 1. Compute raw p-values
    for i, sys_a in enumerate(systems):
        for j, sys_b in enumerate(systems):
            if i >= j: continue # Only upper triangle
            
            col_a = f"{sys_a}_{metric}"
            col_b = f"{sys_b}_{metric}"
            
            if col_a not in df.columns or col_b not in df.columns:
                continue
                
            valid_df = df[[col_a, col_b]].dropna()
            diff = valid_df[col_a] - valid_df[col_b]
            
            # Wilcoxon requires non-zero differences
            diff = diff[diff != 0]
            
            if len(diff) < 5:
                # Too few samples for reliable test
                p_val = 1.0
            else:
                try:
                    # Two-sided test for general difference
                    _, p_val = stats.wilcoxon(valid_df[col_a], valid_df[col_b], alternative='two-sided')
                except ValueError:
                    # Can happen if all differences are zero
                    p_val = 1.0
                    
            pair_key = f"{sys_a} vs {sys_b}"
            p_values[pair_key] = p_val
            comparisons.append((pair_key, p_val))
            
    # 2. Apply Holm-Bonferroni Correction
    # Sort by p-value ascending
    comparisons.sort(key=lambda x: x[1])
    
    m = len(comparisons)
    corrected_p_values = {}
    
    previous_corrected_p = 0.0
    
    for k, (pair_key, raw_p) in enumerate(comparisons):
        # Holm-Bonferroni formula: p_corrected = min(1, max(previous, p * (m - k)))
        # Note: k is 0-indexed here, so rank is k+1. Denominator is m - k.
        
        correction_factor = m - k
        corrected_p = raw_p * correction_factor
        
        # Enforce monotonicity
        corrected_p = max(corrected_p, previous_corrected_p)
        corrected_p = min(corrected_p, 1.0)
        
        corrected_p_values[pair_key] = corrected_p
        previous_corrected_p = corrected_p
        
    return corrected_p_values

def build_comparison_payload(runs_data: List[RunData]) -> Dict:
    """Builds the final aggregated JSON for the visualizer and LLM Synthesizer."""
    if len(runs_data) < 2:
        raise ValueError("Need at least 2 runs to compare.")
    
    joined_df = enforce_ap_rh1(runs_data)
    global_stats = extract_global_stats(runs_data)
    domain_stats = extract_domain_stats(runs_data)
    
    system_names = [r.run_id for r in runs_data]
    
    # Dynamically extract metric intersection across runs
    logger.info("Extracting intersection of available metrics across runs.")
    metric_cols_list = [set(r.results_df.columns) - {"sample_id", "domain", "system_name"} for r in runs_data]
    intersection_metrics = list(set.intersection(*metric_cols_list))
    logger.info(f"Dynamically discovered {len(intersection_metrics)} intersecting metrics: {intersection_metrics}")
    
    # Compute Paired Metrics dynamically
    win_rate_matrix = {}
    pairwise_significance = {}
    for metric in intersection_metrics:
        win_rate_matrix[metric] = compute_win_rates(joined_df, system_names, metric)
        # Note: Depending on the metric, some tests might return p=NaN if the difference is strictly 0.
        # But compute_pairwise_significance handles ties gracefully.
        pairwise_significance[metric] = compute_pairwise_significance(joined_df, system_names, metric)

    # Determine the divergence cases for AP-RH4
    logger.info("Extracting divergence samples (where one system got judge_score=1.0 and another 0.0) for AP-RH4.")
    divergence_samples = []
    
    # We look for binary divergence on judge_score if it exists
    if f"{system_names[0]}_judge_score" in joined_df.columns:
        for _, row in joined_df.iterrows():
            scores = [row[f"{sys}_judge_score"] for sys in system_names]
            if len(set(scores)) > 1 and pd.notna(scores).all():
                divergence_samples.append(row.to_dict())
                
    raw_payload = {
        "metadata": {
            "runs": [r.run_id for r in runs_data],
            "systems": system_names,
            "paired_sample_n": len(joined_df)
        },
        "global_statistics": global_stats,
        "domain_statistics": domain_stats,
        "divergence_pairs_ap_rh4": divergence_samples,
        "win_rate_matrix": win_rate_matrix,
        "pairwise_significance": pairwise_significance,
        # We need to save the joined DF as a CSV for the visualizer to plot distributions
        "joined_dataframe": joined_df.to_dict(orient="records")
    }
    
    # Validate via Pydantic struct before returning
    try:
        validated_payload = AggregatedData(**raw_payload)
        return validated_payload.model_dump()
    except Exception as e:
        logger.error(f"Failed to validate generated research payload against AggregatedData schema: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Research Harness Data Aggregator.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run IDs to compare.")
    parser.add_argument("--runs-dir", default="runs", help="Base directory for runs.")
    parser.add_argument("--out", default="research_insights/aggregated_data.json", help="Output JSON path.")
    args = parser.parse_args()
    
    try:
        runs_data = load_runs(args.runs, args.runs_dir)
        payload = build_comparison_payload(runs_data)
        
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
            
        # Also write the dataframe to a CSV for easy ingestion by visualizer
        df = pd.DataFrame(payload["joined_dataframe"])
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Successfully aggregated {len(args.runs)} runs. Data written to {args.out} and {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to aggregate runs: {e}")
        raise

if __name__ == "__main__":
    main()
