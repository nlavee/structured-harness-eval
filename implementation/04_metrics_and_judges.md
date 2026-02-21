# Phase 4: Metrics & Judge Logic

## Objectives
- Implement Metric Registry.
- Implement Deterministic Metrics (`exact_match`, `soft_recall`, etc.).
- Implement Judge Logic (`EqualityJudge`, `HallucinationJudge`).
- Implement Judge Strategy (`Rotation`).
- Test metrics.

## Plan
1. **Metric Registry**: `glass/metrics/registry.py`. ✅
2. **Metrics**:
   - `exact_match.py` ✅
   - `soft_recall.py` (whitespace tokenization) ✅
   - `verbosity.py` ✅
   - `latency.py` ✅
   - `refusal.py` ✅
   - `error_rate.py` ✅
3. **Judges**:
   - `glass/judges/equality.py` (Merged into `llm.py`) ✅
   - `glass/judges/hallucination.py` (Merged into `llm.py`) ✅
   - `glass/judges/llm.py` ✅
4. **Strategy**: `glass/judges/strategies/rotation.py`. ✅
5. **Tests**: `tests/test_metrics.py`. ✅

## Status
✅ Phase 4 Complete. Tests passed.
