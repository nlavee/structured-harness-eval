from typing import List, Literal

import numpy as np
from scipy import stats


def wilcoxon_test(
    x: List[float],
    y: List[float],
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> float:
    """
    Perform Wilcoxon signed-rank test.
    Returns p-value.
    """
    x = np.array(x)
    y = np.array(y)

    # Filter valid pairs?
    # Standard wilcoxon requires same length.
    if len(x) != len(y):
        raise ValueError("Wilcoxon requires paired data (same length).")

    diff = x - y
    if np.all(diff == 0):
        # All ties
        return 1.0

    res = stats.wilcoxon(x, y, alternative=alternative, method="auto")
    return float(res.pvalue)
