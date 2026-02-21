from typing import List

import numpy as np


def rank_biserial(x: List[float], y: List[float]) -> float:
    """
    Calculate rank-biserial correlation for paired samples.
    r = (sum of positive ranks - sum of negative ranks) / sum of all ranks
    """
    x = np.array(x)
    y = np.array(y)
    diff = x - y

    # Remove zero diffs for ranking? Scipy keeps them usually or handles them.
    # Simple formula:
    # Rank absolute differences.
    # Signed ranks.

    # Implementation using scipy ranks
    from scipy.stats import rankdata

    # Remove zeros for effect size?
    # Common practice is to exclude ties or use specific tie-breaking.
    # Simple implementation:

    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0

    ranks = rankdata(np.abs(diff))

    pos_sum = np.sum(ranks[diff > 0])
    neg_sum = np.sum(ranks[diff < 0])
    total_sum = np.sum(ranks)

    return (pos_sum - neg_sum) / total_sum
