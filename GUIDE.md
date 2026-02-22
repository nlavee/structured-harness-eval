# GLASS Implementation Guide

This guide provides a comprehensive reference for setting up, building, testing, and running the GLASS evaluation harness.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+
- `git`
- CLI tools installed and in `$PATH`:
  - `claude` (Claude Code)
  - `gemini` (Gemini CLI)
  - `codex` (Codex CLI)
  - `structured-harness` (Structured Execution Harness)

### Virtual Environment
It is highly recommended to run GLASS in a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Dependencies
```bash
pip install --upgrade pip
pip install pydantic pyyaml pandas numpy scipy datasets nltk click rich anthropic google-generativeai openai tabulate python-dotenv huggingface_hub pytest
```

### API Keys
Set the following environment variables before running evaluations:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIzaSy..."
export OPENAI_API_KEY="sk-..."
```

---

## 2. Installation & Build

### Editable Install (development)
```bash
pip install -e .
```

### AA-LCR Dataset Setup

GLASS requires the AA-LCR dataset files to be available locally. Run the download script to fetch and set up the data automatically:

```bash
python scripts/download_aalcr.py
```

This downloads the dataset from [HuggingFace](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR), extracts the documents, fixes any filename encoding issues, and verifies all files. The resulting structure:
```
data/AA-LCR/
├── AA-LCR_Dataset.csv
├── AA-LCR_extracted-text/
│   └── lcr/
│       ├── Academia/
│       ├── Company_Documents/
│       └── ...
└── README.md
```

To download to a custom location, use `--dest`:
```bash
python scripts/download_aalcr.py --dest /path/to/data
```

You can customize the dataset path in your config via the `dataset_folder` option:
```yaml
dataset:
  name: "aa_lcr"
  dataset_folder: "data/AA-LCR/AA-LCR_extracted-text/lcr"  # path to extracted text
```

If the local data is not found, GLASS will exit early with a clear error message.

### Build Distribution
```bash
pip install build
python3 -m build
```

---

## 3. Testing

GLASS uses `pytest`. All tests are in `tests/`.

### Run All Tests
```bash
pytest
# or: pytest -v for verbose output
```

### Test Suites

| File | What it covers |
|------|----------------|
| `test_scaffolding.py` | Config schemas, Pydantic models, ABCs, `RawOutput.command` (AP-8) |
| `test_dataset.py` | Registry lookup, AA-LCR adapter mock load |
| `test_systems.py` | Claude + Structured Harness command construction |
| `test_systems_extended.py` | Gemini, Codex, Stub; encoding safety (AP-27); timeout handling |
| `test_metrics.py` | Happy-path for all metrics |
| `test_metrics_edge_cases.py` | AP-3 (None on error), AP-14 (normalisation), AP-15 (undefined cases) |
| `test_pipeline.py` | End-to-end mocked pipeline flow |
| `test_statistics.py` | Bootstrap CI, Wilcoxon, rank-biserial basic |
| `test_statistics_extended.py` | Seeding (AP-23), edge cases, full statistics.json generation |
| `test_human_eval.py` | Cohen's Kappa, compute_agreement, CSV import validation |
| `test_error_analysis.py` | Error crosstab, divergence cases, write report |

**Total: 86 tests, all passing.**

---

## 4. Running Evaluations

The `glass` CLI is the entry point for all operations.

### Smoke Test (no API keys needed — uses stub system)
```bash
glass run configs/smoke_test.yaml
```

### Stratified Subset (20 samples, fast)
```bash
glass run configs/aa_lcr_subset.yaml
```

### Full Evaluation (100 samples, all 6 systems)
```bash
glass run configs/aa_lcr_full.yaml
```

### Model Ablation (Structured Harness across 3 backends)
```bash
glass run configs/aa_lcr_ablation.yaml
```

### Resume an Interrupted Run
```bash
glass run configs/aa_lcr_full.yaml --resume 20260221_120000_runtime_comparison_v1
```

### Task Runner (Alternative)
You can also use the `robo` task runner for simplified command execution:

```bash
# Run evaluation
robo eval:run configs/aa_lcr_subset.yaml

# Launch research harness
robo research:harness --runs <run_id_1> <run_id_2>

# View all tasks
robo help
```

---

## 5. CLI Commands

### `glass run <config>`
Runs the full evaluation pipeline (Phases 0–4).

```
glass run configs/aa_lcr_full.yaml [--resume <run_id>]
```

**Phases:**
1. **Setup** — Creates run directory, writes `manifest.json`, freezes `config.yaml`
2. **Inference** — Calls each SUT on each sample; saves `RawOutput` JSONs; resumable
3. **Statistics** — Computes bootstrap CIs, Wilcoxon tests, effect sizes → `statistics.json`
4. **Reporting** — Generates `results.csv` and `summary.md`

### `glass stats <run_id>`
Recomputes `statistics.json` for a completed run without re-running inference or evaluation.
Also computes Cohen's Kappa if human labels have been imported.

```bash
glass stats 20260221_120000_runtime_comparison_v1
```

### `glass export-human-eval <run_id>`
Exports a stratified sample for human annotation. Guarantees ≥ 2 samples per domain.

```bash
glass export-human-eval 20260221_120000_runtime_comparison_v1 --fraction 0.3
```

Output: `runs/<run_id>/human_eval/export_<date>.csv`

CSV columns: `sample_id, domain, question, gold_answer, system_name, prediction, automated_judge_score, human_label`

Annotators fill in `human_label` (1 = CORRECT, 0 = INCORRECT). All other columns are read-only reference.

### `glass import-human-eval <run_id> <labels.csv>`
Imports completed human labels, validates all rows are filled, attaches labels to `EvalResult` objects, and computes Cohen's Kappa.

```bash
glass import-human-eval 20260221_120000_runtime_comparison_v1 runs/20260221_120000_runtime_comparison_v1/human_eval/export_20260221.csv
```

Kappa is appended to `statistics.json` under `human_eval_agreement`.

---

## 6. Artifacts Produced per Run

```
runs/
└── {run_id}/
    ├── manifest.json          # Git hash, library versions, config SHA256, timestamp, seed
    ├── config.yaml            # Frozen config (canonical source of truth — AP-22)
    ├── glass.log              # Full pipeline log (console + file)
    │
    ├── inference/             # Phase 2: Raw SUT outputs (cached; supports resume)
    │   ├── claude-code/
    │   │   ├── sample_1.json  # RawOutput: prompt, output, latency, exit_code, stderr, command (AP-8)
    │   │   └── checkpoint.json
    │   ├── structured_harness_claude/
    │   └── ...
    │
    ├── evaluation/            # Phase 2: Per-sample metric scores
    │   ├── claude-code/
    │   │   └── sample_1.json  # EvalResult: metrics dict, judge_outputs (AP-12)
    │   └── ...
    │
    ├── human_eval/            # Phase 5 (optional)
    │   ├── export_<date>.csv  # Exported for human annotation
    │   └── labels_<date>.csv  # Imported human labels
    │
    ├── results.csv            # Flat table: one row per (sample × system)
    ├── summary.md             # Aggregate + per-domain tables + statistical highlights
    └── statistics.json        # CIs, p-values, effect sizes, kappa (AP-17, AP-18)
```

## 7. Research Harness (Post-Run Analysis)

For PhD-level evaluation of multiple configurations, use the Research Harness tools located in `research_harness/` to enforce the scientific anti-patterns defined in `research_harness/RESEARCH.md`.

### Orchestrator

The orchestrator will prompt you to select multiple runs to perform paired statistical aggregations and LLM synthesis.
To use it, run:
```bash
python3 research_harness/cli.py
```
You can also bypass the menu:
```bash
python3 research_harness/cli.py --runs RUN_ID_1 RUN_ID_2 --provider gemini --model gemini-2.5-pro
```

### Generated Insights
The harness populates the `research_insights/` directory with timestamped runs (e.g. `run_YYYYMMDD_HHMMSS/`):
- `command.txt`: The exact CLI command used to orchestrate the run.
- `figures/`: Distribution-aware plots, Confidence Interval overlaps, and Domain Performance Heatmaps.
- `logs/`: Chain-of-thought logging for the LLM Synthesizer.
- `aggregated_data.json` & `aggregated_data.csv`: Paired inner-join data, rigorously typed via Pydantic `AggregatedData` schemas.
- `insights_{timestamp}.md`: LLM qualitative synthesis grounded in paired divergence examples and per-domain accuracy tradeoffs.

---

## 8. Directory Structure Reference

```
glass/
├── cli.py                  # CLI entry point (run, stats, export-human-eval, import-human-eval)
├── pipeline.py             # Orchestration: Phases 0–4
├── tui.py                  # Rich-based Terminal User Interface and dual logging
├── config/
│   └── schema.py           # Pydantic models: Config, SystemConfig, JudgeConfig, etc.
├── datasets/
│   ├── base.py             # EvaluationSample, ConversationTurn, DatasetAdapter ABC
│   ├── registry.py         # @register("name") decorator
│   └── aalcr.py            # AA-LCR adapter (local CSV + extracted documents)
├── systems/
│   ├── base.py             # RawOutput (with command field), SystemUnderTest ABC
│   ├── registry.py
│   ├── claude.py           # ClaudeSystem: ["claude", "--print", "--output-format", "text"]
│   ├── gemini.py           # GeminiSystem: ["gemini", "-p"]
│   ├── codex.py            # CodexSystem: ["codex", "exec", "--model", ...]
│   ├── structured_harness.py # StructuredHarnessSystem: ["structured-harness", config.json, "-", "--model", ...]
│   ├── stub.py             # StubSystem: returns gold_answer (for testing)
│   └── structured_configs/
│       └── single_turn_qa.json
├── judges/
│   ├── base.py             # EvalResult, Judge ABC
│   ├── llm.py              # LLMJudge (OpenAI/Anthropic/Google), JudgeAPIError
│   ├── prompts.py          # Versioned prompt constants (AP-11)
│   └── strategies/
│       ├── base.py         # JudgeStrategy ABC
│       └── rotation.py     # RotationStrategy: model-aware, no self-judging (AP-10)
├── metrics/
│   ├── base.py             # BaseMetric ABC
│   ├── registry.py
│   ├── exact_match.py      # Normalised string equality (AP-14)
│   ├── soft_recall.py      # Token-level recall (whitespace tokenisation)
│   ├── soft_f1.py          # Token-level F1 score
│   ├── latency.py          # Wall-clock subprocess time (AP-9)
│   ├── verbosity.py        # len(prediction) / len(gold_answer)
│   ├── refusal.py          # Regex-based refusal detection
│   ├── error_rate.py       # exit_code != 0 or error_type is not None
│   └── utils.py            # normalize_answer() for AP-14
├── statistics/
│   ├── bootstrap.py        # compute_ci(data, n_resamples, alpha, seed) — AP-23
│   ├── significance.py     # wilcoxon_test(x, y, alternative) — AP-18
│   └── effect_size.py      # rank_biserial(x, y)
├── storage/
│   ├── run_store.py        # RunStore: save/load RawOutput + EvalResult
│   ├── manifest.py         # Git hash, lib versions, config SHA256
│   └── checkpoint.py       # Resume support: mark_complete / is_complete
├── human_eval/
│   ├── exporter.py         # Stratified export → human_eval_export.csv
│   ├── importer.py         # Import labels, validate, attach to EvalResult
│   └── agreement.py        # Cohen's Kappa (human vs automated judge)
├── analysis/
│   └── error_taxonomy.py   # ErrorAnalyser: crosstab, divergence, scatter data
└── reports/
    ├── csv_writer.py        # results.csv (per-sample rows)
    ├── summary.py           # summary.md (aggregate + per-domain + AP-19 caveat)
    └── statistics_report.py # statistics.json (CIs + p-values + effect sizes + MC note)

configs/
├── smoke_test.yaml          # 2 samples, stub system (no API keys needed)
├── aa_lcr_full.yaml         # 100 samples, all 6 systems, rotation judge
├── aa_lcr_subset.yaml       # 20 stratified samples, 2 systems, fixed judge
└── aa_lcr_ablation.yaml     # 100 samples, 3 Structured Harness variants, fixed judge
```

---

## 8. Adding New Components

### New Dataset Adapter
```python
# glass/datasets/my_dataset.py
from glass.datasets.base import DatasetAdapter, EvaluationSample
from glass.datasets.registry import register

@register("my_dataset")
class MyDatasetAdapter(DatasetAdapter):
    def load(self): ...
    def get_samples(self) -> list[EvaluationSample]: ...
```
Then add `"my_dataset"` to `dataset.name` in config. No other changes needed.

### Adding a New Metric
1. Create `glass/metrics/my_metric.py`
2. Subclass `BaseMetric` and use `@register`:
   ```python
   from glass.metrics.registry import register
   from glass.metrics.base import BaseMetric
   from glass.systems.base import RawOutput
   from glass.datasets.base import EvaluationSample

   @register("my_metric")
   class MyMetric(BaseMetric):
       def compute(self, output: RawOutput, sample: EvaluationSample, **kwargs) -> float:
           if output.error_type:
               return None
           return len(output.output)
   ```
3. The metric will be automatically discovered because `glass/metrics/__init__.py` auto-imports all metrics in the directory.
4. Add it to your config's `metrics` list. No orchestrator changes needed.

### New System
```python
# glass/systems/my_system.py
from glass.systems.base import SystemUnderTest, RawOutput
from glass.systems.registry import register

@register("my_system")
class MySystem(SystemUnderTest):
    def generate(self, sample) -> RawOutput:
        command = self.config.command or ["my-cli", "--flag"]
        return self._run_command(command, sample.context_prompt, sample.sample_id)
        # _run_command handles: stdin delivery (AP-6), no shell=True (AP-5),
        # UTF-8 decode (AP-27), timeout, stderr, exit code, command field (AP-8)
```

---

## 9. Anti-Pattern Quick Reference

The most common implementation mistakes and how GLASS prevents them:

| Pattern | Correct GLASS Approach |
|---------|------------------------|
| **AP-3** Metric failure → 0.0 | Return `None`; pipeline records `None` in `EvalResult.metrics` |
| **AP-5** `shell=True` subprocess | `SystemUnderTest._run_command()` always uses list args |
| **AP-6** Large CLI args | Prompt passed via `subprocess.communicate(input=prompt.encode())` |
| **AP-8** Command not recorded | `RawOutput.command: List[str]` is required; `prompt` is also saved |
| **AP-10** Self-judging | `RotationStrategy` maps system family → different provider |
| **AP-11** Mutable judge prompts | Constants in `glass/judges/prompts.py`, versioned (V1) |
| **AP-14** No normalisation | `normalize_answer()` in `metrics/utils.py` |
| **AP-15** HR on error output | Pipeline checks `raw_output.error_type`; if present, skips judges and sets metrics to `None` |
| **AP-17** Means without CIs | `summary.md` always shows bootstrap 95% CI |
| **AP-18** t-test on binary scores | Wilcoxon signed-rank (scipy) throughout |
| **AP-19** Per-domain significance | Caveat in `summary.md`; no Wilcoxon per-domain |
| **AP-22** Config edited between phases | Pipeline loads frozen `config.yaml` from run dir |
| **AP-23** No random seed | `bootstrap.compute_ci(seed=config.experiment.seed)` |
| **AP-26** Silent broad exceptions | `JudgeAPIError` raised, logged with traceback, metric → `None` |
| **AP-27** Assume UTF-8 subprocess | `stdout_bytes.decode("utf-8", errors="replace")` |

---

For the full specification and research hypothesis, see **[PLAN.md](./PLAN.md)**.
For detailed notes on each implementation phase, see **[implementation/](./implementation/)**.
