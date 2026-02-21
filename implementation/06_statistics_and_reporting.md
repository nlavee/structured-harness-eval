# Phase 6: Statistics & Reporting

## Objectives
- Implement Bootstrap Confidence Intervals.
- Implement Wilcoxon Signed-Rank Test (One-tailed/Two-tailed).
- Implement Effect Size (Rank-Biserial).
- Implement Markdown Report Generator.
- Integrate into Pipeline.

## Plan
1. **Statistics**:
   - `glass/statistics/bootstrap.py`: `compute_ci(data, n_resamples)` ✅
   - `glass/statistics/significance.py`: `wilcoxon_test(x, y, alternative)` ✅
   - `glass/statistics/effect_size.py`: `rank_biserial(x, y)` ✅
2. **Reporting**:
   - `glass/reports/summary.py`: Generate `summary.md`. ✅
   - `glass/reports/csv_writer.py`: Generate `results.csv`. ✅
3. **Tests**: `tests/test_statistics.py`. ✅

## Status
✅ Phase 6 Complete. Tests passed.
