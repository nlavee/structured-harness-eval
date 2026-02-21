from glass.statistics.bootstrap import compute_ci
from glass.statistics.effect_size import rank_biserial
from glass.statistics.significance import wilcoxon_test


def test_bootstrap():
    data = [1.0] * 50 + [0.0] * 50  # Mean 0.5
    low, high = compute_ci(data, n_resamples=1000)
    # Relax bounds slightly for stochasticity
    assert 0.35 <= low <= 0.55
    assert 0.45 <= high <= 0.65


def test_wilcoxon():
    # Identical
    x = [1, 2, 3]
    y = [1, 2, 3]
    assert wilcoxon_test(x, y) == 1.0

    # Distinct (N=6 needed for p < 0.05 two-sided)
    # 2^-6 = 0.015625 * 2 = 0.03125
    x = [10] * 6
    y = [1] * 6
    p = wilcoxon_test(x, y)
    assert p < 0.05


def test_rank_biserial():
    # All x > y
    x = [2, 3, 4]
    y = [1, 2, 3]
    # diffs = [1, 1, 1]. All pos.
    assert rank_biserial(x, y) == 1.0

    # All x < y
    x = [1, 2, 3]
    y = [2, 3, 4]
    # diffs = [-1, -1, -1]. All neg.
    assert rank_biserial(x, y) == -1.0

    # Mixed
    x = [2, 1]
    y = [1, 2]
    # diffs = [1, -1].
    # abs diffs = [1, 1]. Ranks = [1.5, 1.5]
    # pos_sum = 1.5, neg_sum = 1.5
    # (1.5 - 1.5) / 3 = 0
    assert rank_biserial(x, y) == 0.0
