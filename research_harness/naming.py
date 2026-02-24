from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

class PlotType(Enum):
    FOREST = "forest_ci"
    VIOLIN = "violin_dist"
    PAIRED_DIFF = "paired_diff"
    WIN_RATE = "win_rate"
    SIGNIFICANCE = "significance"
    DOMAIN_HEATMAP = "domain_heatmap"
    RADAR = "radar_behavior"

def get_plot_filename(plot_type: PlotType, metric: Optional[str] = None) -> str:
    """Generates a filename for a given plot type and metric."""
    if plot_type == PlotType.RADAR:
        return f"{plot_type.value}.png"
    if not metric:
        raise ValueError(f"Metric must be provided for plot type {plot_type.name}")
    return f"{plot_type.value}_{metric}.png"

def parse_plot_filename(filename: str) -> Tuple[Optional[PlotType], Optional[str]]:
    """Parses a filename to extract the plot type and metric."""
    if filename == "radar_behavior.png":
        return PlotType.RADAR, "Behavioral Profile"
    
    if not filename.endswith(".png"):
        return None, None
    
    name = filename[:-4]  # Remove .png
    
    for pt in PlotType:
        if pt == PlotType.RADAR:
            continue
        prefix = f"{pt.value}_"
        if name.startswith(prefix):
            metric = name[len(prefix):]
            return pt, metric
            
    return None, None
