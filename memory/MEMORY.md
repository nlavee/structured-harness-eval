# GLASS Project Memory

## Project: structured-harness-eval / GLASS evaluation harness

### Architecture
- Python package at `glass/`, installed via `pip install -e .`
- Registry pattern: `@register("name")` decorators for datasets, systems, metrics
- 5-phase pipeline: Setup → Inference → Evaluation → Statistics → Reporting
- Stats and reporting strictly after all samples processed (AP-4)

### Key Files
- `glass/pipeline.py`: main orchestration
- `glass/cli.py`: `glass run | stats | export-human-eval | import-human-eval`
- `glass/reports/statistics_report.py`: Phase 3, generates `statistics.json`
- `glass/human_eval/`: exporter, importer, agreement (Cohen's Kappa)
- `glass/analysis/error_taxonomy.py`: qualitative error analysis
- `configs/`: `smoke_test.yaml`, `aa_lcr_full.yaml`, `aa_lcr_subset.yaml`, `aa_lcr_ablation.yaml`

### Test setup
- pytest, 86 tests, all passing
- Run: `python3 -m pytest tests/ -v`
- No `venv` by default — uses system Python3 with `--break-system-packages`

### Critical patterns
- `RawOutput` requires `command: List[str]` field (AP-8 compliance)
- Metrics return `None` (not 0.0) on error or undefined cases (AP-3, AP-15)
- Bootstrap CI uses `seed=` parameter for reproducibility (AP-23)
- `JudgeAPIError` raised on LLM failures; pipeline catches and sets metric=`None`
- Evaluation dir uses `evaluation/{system_name}/sample_{id}.json` (not flat)
