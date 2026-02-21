from pathlib import Path
from typing import List

import pandas as pd

from glass.judges.base import EvalResult


def write_results_csv(results: List[EvalResult], output_path: Path) -> None:
    data = []
    for r in results:
        row = {
            "sample_id": r.sample_id,
            "system_name": r.system_name,
            "domain": r.domain,
        }
        # Flatten metrics
        for k, v in r.metrics.items():
            row[k] = v
        # Add judge explanations if useful? Maybe too verbose for main CSV.
        # Just metrics for now.
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
