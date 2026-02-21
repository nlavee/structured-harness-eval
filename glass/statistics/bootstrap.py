from typing import List, Optional, Tuple

import numpy as np


def compute_ci(
    data: List[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute bootstrap (1-alpha) confidence interval for the mean.

    Args:
        data: Observed values (NaN/None already filtered by caller).
        n_resamples: Number of bootstrap resamples (default 10,000 per PLAN.md).
        alpha: Significance level; returns (alpha/2, 1-alpha/2) percentile CI.
        seed: Optional RNG seed for reproducibility (AP-23).

    Returns:
        (lower, upper) bounds of the CI.
    """
    if not data:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    arr = np.array(data, dtype=float)
    # Vectorised: shape (n_resamples, n) → mean per resample
    resampled_means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)

    lower = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    upper = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return lower, upper
