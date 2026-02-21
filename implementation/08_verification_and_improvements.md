# Phase 8: Verification and Improvements

## Performed by: Claude Sonnet 4.6 (verification pass)
## Date: 2026-02-21

---

## 1. Verification Summary

A full verification pass was performed against `PLAN.md`, including:
- Anti-pattern compliance audit (all 27 anti-patterns)
- Feature completeness vs. PLAN spec
- Test coverage analysis
- Code quality review

**Result:** 18 PASS, 8 PARTIAL, 3 FAIL before fixes. After fixes: all critical issues resolved.

---

## 2. Issues Found and Fixed

### 2.1 AP-8 VIOLATION — `RawOutput` missing `command` field (CRITICAL)

**Problem:** `glass/systems/base.py`'s `RawOutput` Pydantic model had no `command: List[str]` field.
Per PLAN.md AP-8, the exact command invoked must be stored for reproducibility audits.

**Fix:** Added `command: List[str]` as a required field to `RawOutput`. Updated all three
construction sites in `_run_command()` (normal path, timeout path, exception path) to
populate it. Updated `StubSystem` which constructs `RawOutput` directly. Updated all tests.

---

### 2.2 Missing Statistics Phase (CRITICAL — AP-4 VIOLATION)

**Problem:** `pipeline.py` had no Phase 3 statistics computation. `glass/statistics/`
module existed but was completely unused by the pipeline. `statistics.json` was never
generated. This violated AP-4 ("do not compute statistics inline") and PLAN.md Phase 3.

**Fix:**
- Created `glass/reports/statistics_report.py` implementing the full Phase 3 spec:
  - Per-system aggregate stats (mean, std, median) with bootstrap 95% CIs
  - Primary hypothesis: one-tailed Wilcoxon for `judge_score` (direction: "greater")
    and `hallucination_rate` (direction: "less") — pre-specified, not chosen post-hoc
  - Secondary metrics: two-tailed Wilcoxon for all other metrics
  - Rank-biserial effect sizes for all comparisons
  - Per-domain descriptive stats (no significance tests, per AP-19)
  - Multiple-comparisons acknowledgment metadata (AP-20)
- Updated `pipeline.py` to call Phase 3 statistics before reporting

---

### 2.3 Missing CLI Commands (MAJOR)

**Problem:** `glass/cli.py` only had `glass run`. PLAN.md specifies three additional commands:
- `glass stats <run_id>`
- `glass export-human-eval <run_id>`
- `glass import-human-eval <run_id> <labels.csv>`

**Fix:** Implemented all three commands in `cli.py` with full argument handling, validation,
and proper frozen config loading (AP-22).

---

### 2.4 Missing Human Evaluation Module (MAJOR)

**Problem:** `glass/human_eval/` directory was empty. PLAN.md specifies three components.

**Fix:** Created:
- `glass/human_eval/exporter.py`: Stratified sampling (min 2 per domain), exports CSV with
  blank `human_label` column. Respects config seed for reproducibility.
- `glass/human_eval/importer.py`: Validates completed CSVs, checks all labels are filled
  (0 or 1), attaches `human_label` to `EvalResult` and persists updates.
- `glass/human_eval/agreement.py`: Cohen's Kappa implementation (formula-based, not via
  sklearn to avoid dependency). Computes per-system and overall κ. Returns disagreement
  rows for qualitative review (flagged in summary per PLAN.md).

---

### 2.5 Missing Error Analysis Module (MAJOR)

**Problem:** `glass/analysis/error_taxonomy.py` referenced in PLAN.md did not exist.

**Fix:** Created the module with:
- `error_type_crosstab()`: system × domain × error_type counts
- `divergence_cases()`: samples where Structured Harness succeeds and baseline fails (and vice versa)
- `verbosity_vs_correctness()`: raw rows for scatter analysis
- `hallucination_by_domain()`: mean HR per domain × system
- `run()`: full report dict
- `write_report()`: JSON + Markdown output

---

### 2.6 Bootstrap Not Seeded (AP-23)

**Problem:** `compute_ci()` called `np.random.choice` without a seed. CI values were
non-deterministic across runs.

**Fix:** Added `seed: Optional[int] = None` parameter to `compute_ci()`. Uses
`np.random.default_rng(seed)` (modern API). All callers in `statistics_report.py` and
`summary.py` pass `config.experiment.seed`.

---

### 2.7 Codex System Ignoring `config.model` (MINOR)

**Problem:** `glass/systems/codex.py` hardcoded `--model gpt-5` and silently ignored
`self.config.model`. If someone specified `model: "gpt-4.1"` in the config, it was ignored.

**Fix:** Codex now uses `self.config.model or "gpt-5"` when building the default command.
Config override via `command:` still respected.

---

### 2.8 `SoftRecallMetric` Redundant Check

**Problem:** Lines 17 and 23 both checked `if not gold_tokens: return 0.0`. The second
check was unreachable. Also, returning `0.0` for empty gold is semantically wrong — it
should be `None` (undefined), following AP-3 and AP-15 principles.

**Fix:** Removed duplicate check; changed return value to `None` for empty gold.

---

### 2.9 `VerbosityMetric` Returning 0.0 for Undefined Case

**Problem:** `verbosity.py` returned `0.0` when gold_answer is empty. This is semantically
a 0.0 verbosity score, but it actually means "undefined" (can't compute a ratio).
Per AP-3: metric failures and undefined cases must be `None`, not silent zeros.

**Fix:** Returns `None` when `gold_len == 0`.

---

### 2.10 LLM Judge Error Handling (AP-26)

**Problem:** `LLMJudge._call_llm()` caught all exceptions and returned `"ERROR: ..."` string.
The caller in `pipeline.py` had no way to distinguish a real judge response from an error
string. An error string would flow into metric computation as if it were judge output,
producing silent incorrect scores.

**Fix:**
- Created `JudgeAPIError` exception class in `glass/judges/llm.py`
- `_call_llm()` now raises `JudgeAPIError` on any API failure (with full traceback logged)
- `pipeline.py` Phase 2 catches `JudgeAPIError` and sets metric to `None` (not 0.0)
- Stores `"ERROR: ..."` in `judge_outputs` for audit trail (AP-12 preserved)

---

### 2.11 Metrics Reinstantiated Per Sample (Performance)

**Problem:** `pipeline.py` created `metric_map = {name: get_metric_class(name)() ...}` inside
the per-sample loop, creating new metric instances for every sample × system combination.

**Fix:** Moved metric instantiation outside both loops (once per pipeline run). Metrics are
stateless (AP-25), so this is safe and more efficient.

---

### 2.12 Missing Per-Domain Caveats (AP-19)

**Problem:** `summary.md` per-domain breakdown had no note about insufficient statistical
power (~14 samples per domain).

**Fix:** Added AP-19 caveat: "Per-domain samples are typically N≈14, which is insufficient
for statistical significance claims. These numbers are descriptive only."

---

### 2.13 Missing Config Templates

**Problem:** Only `smoke_test.yaml` existed. PLAN.md specifies three templates.

**Fix:** Created:
- `configs/aa_lcr_full.yaml`: All 100 samples, all 6 systems, rotation judge
- `configs/aa_lcr_subset.yaml`: 20 stratified samples, 2 systems, fixed judge
- `configs/aa_lcr_ablation.yaml`: All samples, 3 Structured Harness variants, fixed judge

---

## 3. Anti-Pattern Compliance After Fixes

| # | Anti-Pattern | Before | After |
|---|---|---|---|
| AP-1 | Inference/eval mixing | ✅ | ✅ |
| AP-2 | Raw outputs not saved | ✅ | ✅ |
| AP-3 | Metric failure → 0.0 | ⚠️ | ✅ (soft_recall, verbosity fixed; judge error handling fixed) |
| AP-4 | Inline statistics | ❌ | ✅ (Phase 3 statistics added) |
| AP-5 | shell=True | ✅ | ✅ |
| AP-6 | Large CLI args | ✅ | ✅ |
| AP-7 | Ignoring stderr/exit | ✅ | ✅ |
| AP-8 | Command not recorded | ❌ | ✅ (command field added) |
| AP-9 | Latency mislabelled | ✅ | ✅ |
| AP-10 | Self-judging | ✅ | ✅ |
| AP-11 | Unversioned judge prompts | ✅ | ✅ |
| AP-12 | Discarding judge text | ✅ | ✅ |
| AP-13 | Determinism assumption | ✅ | ✅ |
| AP-14 | No normalisation | ✅ | ✅ |
| AP-15 | HR on error outputs | ✅ | ✅ |
| AP-16 | Wrong/refused conflated | ⚠️ | ✅ (separate metrics) |
| AP-17 | Means without CIs | ✅ | ✅ |
| AP-18 | t-test on binary scores | ✅ | ✅ |
| AP-19 | Per-domain significance | ⚠️ | ✅ (caveat added) |
| AP-20 | Multiple comparisons | ❌ | ✅ (MC note in statistics.json) |
| AP-21 | Model version not pinned | ⚠️ | ⚠️ (noted; full resolution requires API integration) |
| AP-22 | Config modified between phases | ✅ | ✅ |
| AP-23 | No random seed | ⚠️ | ✅ (bootstrap now seeded) |
| AP-24 | Metric logic in orchestrator | ✅ | ✅ |
| AP-25 | Mutable metric state | ✅ | ✅ |
| AP-26 | Silent broad exceptions | ⚠️ | ✅ (JudgeAPIError; logged with traceback) |
| AP-27 | Assume UTF-8 subprocess | ✅ | ✅ |

---

## 4. Test Coverage After Improvements

| File | Tests | Coverage |
|---|---|---|
| `test_scaffolding.py` | 5 | Schemas, ABCs, RawOutput.command field |
| `test_dataset.py` | 2 | Registry, AA-LCR mock load |
| `test_systems.py` | 2 | Claude, Structured Harness command construction |
| `test_systems_extended.py` | 9 | Gemini, Codex, Stub, encoding, timeout |
| `test_metrics.py` | 5 | Happy path for all metrics |
| `test_metrics_edge_cases.py` | 22 | AP-3, AP-14, AP-15: None on error, normalisation, undefined cases |
| `test_pipeline.py` | 1 | End-to-end mocked pipeline flow |
| `test_statistics.py` | 3 | Bootstrap, Wilcoxon, rank-biserial basic |
| `test_statistics_extended.py` | 12 | Seeding (AP-23), edge cases, full report generation |
| `test_human_eval.py` | 14 | Cohen's Kappa, agreement, CSV import/validation |
| `test_error_analysis.py` | 7 | Crosstab, divergence, hallucination, write report |
| **Total** | **82** | **All pass** |

---

## 5. Remaining Open Items (Future Work)

### AP-21: Resolved Model Version Logging
The manifest captures library versions but not the resolved model version (e.g., what "sonnet"
maps to at runtime). Full resolution requires:
- Querying each SUT CLI for `--version` output
- Parsing API response metadata for model version fields
- This adds latency to Phase 0 and may not be supported by all CLIs

**Recommendation:** Log CLI `--version` output in manifest as a best-effort field.
Annotate in paper that model aliases are not stable across time.

### Structured Harness Integration Testing
Current tests mock all subprocess calls. An end-to-end smoke test against a real Structured Harness
installation (using `stub` system type) verifies the single_turn_qa.json template works
correctly. This requires Structured Harness to be installed in the test environment.

### Human Eval Exporter: Question/Gold Answer in CSV
The exporter currently stores `question` and `gold_answer` as empty strings because
`EvalResult` doesn't carry them (only `EvaluationSample` does). To include them in the
human eval CSV, the pipeline could either:
a) Inject them as special `_question` / `_gold_answer` keys in `EvalResult.metrics` (hacky)
b) Store a separate `samples.json` in the run directory and cross-reference by sample_id

**Recommendation:** Option (b) — store `EvaluationSample` objects as `samples.json` in
the run directory during Phase 0, load in the exporter by sample_id.

### Structured Harness `single_turn_qa.json` Template
The current template uses `{{input}}` placeholder. If Structured Harness's actual template syntax
differs, this will fail silently (empty output). Add an explicit integration test or
documentation cross-reference to the Structured Harness team's confirmed template format.

---

## 6. Architectural Notes

### Why `evaluation/{system_name}/sample_{id}.json` instead of `evaluation/sample_{id}.json`
PLAN.md specifies a flat `evaluation/sample_{id}.json`, but `EvalResult` is per-(sample,
system). A flat structure would require either:
a) One file per sample containing all systems' results — complicates partial loading
b) Overwriting with the last system's result — loses data

The implementation uses `evaluation/{system_name}/sample_{id}.json` (mirroring inference).
This is a deliberate, documented deviation that preserves all data.

### judge_score and hallucination_rate as "inline" metrics
These are computed in the pipeline's evaluation phase by calling the judge, not as
registry-based metrics. They are added directly to `metric_results` dict. The config's
`metrics:` list should not include `judge_score` or `hallucination_rate` (those would fail
registry lookup). The config templates include a comment noting this.

**Recommendation:** Consider adding `JudgeScoreMetric` and `HallucinationRateMetric` wrappers
in the registry that delegate to the judge strategy. This would unify the metrics interface
and allow users to include/exclude judge-based metrics from the config list.

### Statistics phase computes Wilcoxon only for N≥4 pairs
The implementation skips Wilcoxon tests when there are fewer than 4 paired samples, noting
"Insufficient paired samples." This is conservative but prevents scipy warnings/errors on
degenerate inputs. The threshold of 4 comes from the minimum meaningful Wilcoxon test size.
