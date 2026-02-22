import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import sys

# Add current directory to path so we can import schema
sys.path.append(str(Path(__file__).parent))
from schema import AggregatedData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
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
    
    # Base configuration:
    # Instead of relying on automatic `_x` and `_y` suffixes in pandas merge,
    # we explicitly extract the columns we want for each run and prefix them.
    
    # Start with just the sample_id and domain
    base_df = runs_data[0].results_df[["sample_id", "domain"]].copy()
    joined_df = base_df
    
    metric_cols = [c for c in runs_data[0].results_df.columns if c not in ["sample_id", "domain", "system_name"]]
    
    for run in runs_data:
        df_run = run.results_df.copy()
        
        # We use the run_id as the unique prefix per user request for clarity in visuals
        prefix = run.run_id
        
        rename_mapping = {col: f"{prefix}_{col}" for col in metric_cols}
        
        df_run_metrics = df_run[["sample_id", "domain"] + metric_cols].rename(columns=rename_mapping)
        
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

def build_comparison_payload(runs_data: List[RunData]) -> Dict:
    """Builds the final aggregated JSON for the visualizer and LLM Synthesizer."""
    if len(runs_data) < 2:
        raise ValueError("Need at least 2 runs to compare.")
    
    joined_df = enforce_ap_rh1(runs_data)
    global_stats = extract_global_stats(runs_data)
    domain_stats = extract_domain_stats(runs_data)
    
    system_names = [r.run_id for r in runs_data]
    
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
