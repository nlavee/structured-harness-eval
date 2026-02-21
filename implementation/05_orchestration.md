# Phase 5: Orchestration

## Objectives
- Implement storage (Runs, Manifests, Checkpoints).
- Implement the main execution pipeline.
- Implement CLI entry point.
- Test end-to-end flow (mocked).

## Plan
1. **Storage**:
   - `glass/storage/manifest.py`: Generate manifest.json. ✅
   - `glass/storage/run_store.py`: Save/load raw outputs and results. ✅
   - `glass/storage/checkpoint.py`: Handle resume logic. ✅
2. **Pipeline**: `glass/pipeline.py` (Inference -> Eval). ✅
3. **CLI**: `glass/cli.py`. ✅
4. **Tests**: `tests/test_pipeline.py`. ✅

## Status
✅ Phase 5 Complete. Tests passed.
