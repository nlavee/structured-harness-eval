from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict

class MetricStats(BaseModel):
    n: int
    mean: float
    std: Optional[float] = None
    median: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

class RunMetadata(BaseModel):
    runs: List[str]
    systems: List[str]
    paired_sample_n: int

class AggregatedData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    metadata: RunMetadata
    # Dict[run_id][metric_name] -> MetricStats
    global_statistics: Dict[str, Dict[str, MetricStats]]
    # Dict[run_id][domain_name][metric_name] -> MetricStats
    domain_statistics: Dict[str, Dict[str, Dict[str, MetricStats]]]
    
    # Store divergence cases dynamically
    divergence_pairs_ap_rh4: List[Dict[str, Any]]
    
    # Dump the joined inner-join df for visualizer analysis
    joined_dataframe: List[Dict[str, Any]]
