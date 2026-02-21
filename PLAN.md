# GLASS: Global Long-context Agent Scoring System

> **Name:** GLASS — an open-source, research-grade evaluation harness for comparing AI agent runtimes on long-context reasoning tasks. Inspired by the **Glass Frog**, known for its transparent skin, GLASS is built on a "transparent-box" philosophy where every raw output, intermediate trace, and judge rationale is preserved and exposed for rigorous auditability.

---

## Research Hypothesis

> **Primary claim:** The Structured Execution Harness achieves a higher task success rate and lower hallucination rate than raw CLI runtimes (Claude Code, Gemini CLI) on long-horizon multi-document reasoning tasks.

This is a **directional hypothesis**. The statistical design uses one-tailed tests where the direction is "Structured Harness ≥ baseline" for success rate and "Structured Harness ≤ baseline" for hallucination rate. The paper will also include a qualitative error analysis exploring *why* the harness produces these differences, not just *that* it does.

**Experimental framing note:** The Structured Execution Harness can run on multiple underlying model backends (Claude, Gemini, GPT), making it a harness-level comparison, not a model-level one. System labels always include the backend model:

| System Label | Runtime | Underlying Model |
|---|---|---|
| `claude-code` | Claude Code CLI | Claude (latest) |
| `gemini-cli` | Gemini CLI | Gemini (latest) |
| `codex-cli` | Codex CLI | GPT 5 |
| `structured_harness (claude)` | Structured Execution Harness | Claude (same version) |
| `structured_harness (gemini)` | Structured Execution Harness | Gemini (same version) |
| `structured_harness (gpt)` | Structured Execution Harness | GPT 5 |

Comparing `claude-code` vs. `structured_harness (claude)` isolates the harness effect with the model held constant — this is the core scientific contribution.

---

## Project Goal

Build a modular, open-source evaluation harness that rigorously tests the above hypothesis using the `ArtificialAnalysis/AA-LCR` dataset as the primary benchmark. The system is designed to:

1. Be **reproducible** — any researcher can re-run the full evaluation from a config file.
2. Be **extensible** — new datasets, metrics, judges, and SUTs can be added via plugins without touching core orchestration code.
3. Produce **publication-ready outputs** — statistics, per-domain breakdowns, and human-judge agreement metrics suitable for independent research reports and potential conference submission.

---

## Key Features

- **Two-Phase Pipeline:** Inference (SUT calls) is fully decoupled from Evaluation (metrics + judges). Raw SUT outputs are cached to disk, enabling metric iteration without re-running expensive CLI calls.
- **Artifact-First Logging:** All outputs are saved to a structured `runs/` directory with a unique run ID. Every file path is explicitly logged to stderr so results can be debugged, retraced, and shared.
- **Plugin Registries:** Datasets, metrics, and judges use a `@registry.register("name")` decorator pattern. Adding a new component requires one file with no changes to orchestration code.
- **Model-Aware Judge Rotation:** Judge assignment is aware of the underlying model in each SUT label, ensuring no self-judging. Configurable as `rotation` (each SUT judged by a competitor) or `fixed` (neutral third party).
- **Multi-Turn Architecture:** The `SystemUnderTest` base class and `EvaluationSample` schema support multi-turn agent interactions, ready for future datasets. AA-LCR runs in single-shot mode.
- **Domain-Stratified Sampling:** Subset runs preserve domain proportions across the 7 AA-LCR document categories.
- **Statistical Analysis Module:** Bootstrap confidence intervals, one-tailed Wilcoxon signed-rank tests (primary hypothesis), and rank-biserial effect sizes. All computed as a separate pipeline phase.
- **Human Evaluation Module:** Exports 20–40% of samples for human annotation, imports labels, and computes Cohen's Kappa between human and automated judge. Flags disagreements for qualitative review.
- **Checkpoint & Resume:** Inference phase writes per-sample checkpoints; interrupted runs resume without data loss.
- **Reproducibility Manifest:** Every run captures harness git hash, library versions, config hash, run ID, and timestamp.

---

## Technical Specifications

- **Language:** Python 3.10+
- **Core Libraries:**
    - `pydantic>=2.0` (Schema validation)
    - `pyyaml` (Config parsing)
    - `pandas` (Data manipulation)
    - `numpy` (Bootstrap CI)
    - `scipy` (Significance tests)
    - `datasets` (Hugging Face dataset loading)
    - `nltk` (Sentence splitting for hallucination metric)
    - `click` (CLI interface)
    - `rich` (Pretty logging/console output)
    - `anthropic`, `google-generativeai`, `openai` (Judge/SUT integrations)
- **Runtime Invocation:** `subprocess` with explicit timeout, stdout/stderr capture, and exit code logging. Never use `shell=True` — prompt text may contain shell metacharacters.
- **Prompt Delivery:** AA-LCR prompts are ~400KB of text. Passing this as a CLI argument risks OS `ARG_MAX` limits. All SUTs receive the assembled prompt via **stdin** using `subprocess.communicate(input=prompt.encode())`. The `-p` / `--print` flag triggers non-interactive mode; the full prompt comes from stdin rather than as an argument to the flag.
  - `ClaudeSystem`: `["claude", "--print", "--output-format", "text"]` — prompt via stdin
  - `GeminiSystem`: `["gemini", "-p"]` — prompt via stdin (Gemini's `-p` flag enables non-interactive mode; stdin provides the prompt content)
  - `StructuredHarnessSystem`: `["structured-harness", "config.json", "-", "--model", "<id>", "--headless", "--autostart"]` — explicit `-` data-path signals stdin; treated as equivalent to Claude Code's `--print` mode per team convention
- **Structured Harness Output:** Treated as stdout, equivalent to Claude Code's `--print` mode (per Q1 resolution). The `single_turn_qa.json` single-codon design produces its answer on stdout. No output-file extraction needed.
- **Structured Harness Model Selection:** `--model <id>` flag (e.g., `sonnet`, `gemini-2.0-flash`, `gpt-4.1`).
- **Judge Integration:** `google-generativeai`, `anthropic`, `openai` — optional; only providers referenced in `config.yaml` need to be installed.
- **Output Formats:** `results.csv` (per-sample), `summary.md` (aggregate), `statistics.json` (CIs + p-values), `human_eval_export.csv` (annotation export).

---

## Dataset Profile

### AA-LCR (v1 Benchmark — Full Scope)

| Property | Value |
|---|---|
| Total samples | 100 questions |
| Document sets | 30 |
| Domains | 7 (Academia, Company Reports, Government Consultations, Industry Reports, Legal, Marketing, Survey Reports) |
| Avg context length | ~99,325 tokens per document set (~71k–115k range) |
| Prompt Construction | Documents are loaded according to the source CSV's `data_source_filenames` to ensure correct ordering, encapsulated in BEGIN/END blocks. |
| Official judge | Binary CORRECT / INCORRECT (Qwen3 235B reference) |

**Paper scope:** The v1 paper uses all 100 AA-LCR samples. Per-domain breakdowns (~14 samples each) are **exploratory only** and are explicitly framed as such in the paper. Aggregate cross-system comparisons have adequate statistical power for publication.

### Statistical Power

| Scope | N samples | Primary test | Power (α=0.05, medium effect) |
|---|---|---|---|
| Full AA-LCR (aggregate) | 100 | Wilcoxon signed-rank (paired) | ~0.80 ✓ |
| Per-domain | ~14 | Exploratory / descriptive only | Insufficient — noted in paper |

---

## Architecture Design

### 1. Directory Structure

```
glass/                              # Python package (pip install -e .)
├── __init__.py
├── cli.py                          # `glass run config.yaml`
│                                   #  `glass export-human-eval run_id`
│                                   #  `glass import-human-eval run_id labels.csv`
│                                   #  `glass stats run_id`
│
├── tui.py                          # Rich-based Terminal User Interface
│
├── config/
│   ├── schema.py                   # Pydantic models: ExperimentConfig, SystemConfig, etc.
│   └── templates/
│       ├── aa_lcr_full.yaml        # All 100 samples, rotation judge
│       ├── aa_lcr_subset.yaml      # Stratified subset, fixed judge
│       └── aa_lcr_ablation.yaml    # Model ablation (structured harness variants only)
│
├── datasets/
│   ├── base.py                     # DatasetAdapter ABC, EvaluationSample schema
│   ├── registry.py                 # @registry.register decorator
│   └── aalcr.py                    # AALCRAdapter: HF download, zip extract, prompt template
│
├── systems/
│   ├── base.py                     # SystemUnderTest ABC, RawOutput schema
│   │                               # Supports both single-turn and multi-turn generate()
│   ├── claude.py                   # ClaudeSystem: `claude` (prompt via stdin)
│   ├── gemini.py                   # GeminiSystem: `gemini` (prompt via stdin)
│   ├── codex.py                    # CodexSystem: `codex` (prompt via stdin)
│   ├── structured_harness.py       # StructuredHarnessSystem: `structured-harness config.json - --model <m>` (stdin)
│   └── structured_configs/
│       └── single_turn_qa.json     # Standard config for single-shot QA tasks (AA-LCR)
│
├── judges/
│   ├── base.py                     # Judge ABC, JudgeScore schema
│   ├── registry.py
│   ├── equality.py                 # EqualityJudge: AA-LCR CORRECT/INCORRECT prompt
│   ├── hallucination.py            # HallucinationJudge: NLI per-sentence classification
│   └── strategies/
│       ├── base.py                 # JudgeStrategy ABC
│       ├── fixed.py                # FixedJudgeStrategy: one judge for all SUTs
│       └── rotation.py             # RotationStrategy: model-aware, no self-judging
│
├── metrics/
│   ├── base.py                     # BaseMetric ABC
│   ├── registry.py                 # @registry.register decorator
│   ├── exact_match.py
│   ├── soft_recall.py              # Token-level F1 / entity recall
│   ├── judge_score.py              # Wraps EqualityJudge → 0/1 score
│   ├── hallucination_rate.py       # Wraps HallucinationJudge → HR score
│   ├── latency.py                  # Wall-clock subprocess time
│   ├── verbosity.py                # len(prediction) / len(gold_answer)
│   └── refusal.py                  # Pattern-based refusal detection
│
├── statistics/
│   ├── bootstrap.py                # Bootstrap CI (n=10,000 resamples)
│   ├── significance.py             # Wilcoxon signed-rank (one-tailed for hypothesis)
│   └── effect_size.py              # Rank-biserial correlation
│
├── human_eval/
│   ├── exporter.py                 # Stratified sample export → human_eval_export.csv
│   ├── importer.py                 # Import completed labels → human_labels.csv
│   └── agreement.py                # Cohen's Kappa (human vs. automated judge)
│
├── analysis/
│   └── error_taxonomy.py           # Categorize and summarize errors by type + domain
│                                   # Supports qualitative "why" analysis for paper
│
├── storage/
│   ├── run_store.py                # Save/load per-sample artifacts
│   ├── manifest.py                 # Reproducibility manifest generation
│   └── checkpoint.py              # Checkpoint write/read for resume
│
└── reports/
    ├── csv_writer.py               # results.csv (per-sample rows)
    ├── summary.py                  # summary.md (aggregate + per-domain)
    └── statistics_report.py        # statistics.json + LaTeX table export
```

### 2. Run Artifact Structure

```
runs/
└── {run_id}/                             # e.g., 20260221_143000_runtime_v1
    ├── manifest.json                     # git hash, lib versions, config hash, timestamp
    ├── config.yaml                       # Frozen copy of config used
    │
    ├── inference/                        # Phase 1: Raw SUT outputs (cached)
    │   ├── claude_code/
    │   │   ├── sample_001.json
    │   │   └── checkpoint.json           # Last completed index (for resume)
    │   ├── gemini_cli/
    │   ├── structured_harness_claude/
    │   ├── structured_harness_gemini/
    │   └── structured_harness_gpt/
    │
    ├── evaluation/                       # Phase 2: Per-sample metric scores
    │   └── sample_001.json
    │
    ├── human_eval/                       # Phase 3 (optional): Human annotation
    │   ├── export_20260221.csv           # Exported samples for human review
    │   └── labels_20260225.csv           # Imported human labels
    │
    ├── results.csv                       # Flat table: one row per (sample × SUT)
    ├── summary.md                        # Aggregate + per-domain table
    └── statistics.json                   # CIs, p-values, effect sizes, kappa
```

**Logging convention:** Every file saved emits a structured log line:
```
[GLASS] Saved raw output → runs/20260221_143000_runtime_v1/inference/structured_harness_claude/sample_042.json
```

### 3. Core Data Schemas (Pydantic)

```python
class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class EvaluationSample(BaseModel):
    sample_id: str
    domain: str                         # e.g., "Legal"
    question: str
    gold_answer: str
    context_prompt: str                 # Fully assembled prompt with documents
    input_tokens: int
    turn_type: Literal["single", "multi"] = "single"
    prior_turns: list[ConversationTurn] = []   # Empty for AA-LCR; populated for multi-turn datasets
    metadata: dict                      # document_set_id, data_source_filenames, etc.

class RawOutput(BaseModel):
    sample_id: str
    system_name: str                    # e.g., "structured_harness_claude"
    model: str | None = None            # Logged for reproducibility
    command: list[str]                  # AP-8: exact command invoked
    prompt: str                         # Full prompt sent via stdin for debugging
    output: str
    chain_of_thought: str | None = None # CoT/reasoning captured when quiet=False
    latency_s: float
    exit_code: int
    stderr: str
    error_type: Literal["timeout", "api_error", "refusal", "malformed", "crash"] | None
    timestamp: str

class EvalResult(BaseModel):
    sample_id: str
    system_name: str
    domain: str
    judge_model: str | None = None      # Judge provider/model metadata
    metrics: dict[str, float]           # e.g., {"judge_score": 1.0, "hallucination_rate": 0.0}
    judge_outputs: dict[str, str]       # Raw judge text responses (for auditability)
    human_label: int | None = None      # 0/1 if this sample was spot-checked
```

### 4. Plugin Registry Pattern

```python
# Shared pattern used by datasets/, metrics/, judges/
_REGISTRY: dict[str, type] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator
```

Adding a new metric requires:
1. One new file in `metrics/` with `@registry.register("my_metric")`
2. Adding `"my_metric"` to `config.yaml` under `metrics:`

Zero changes to orchestration code.

### 5. Structured Harness System Invocation

```python
# systems/structured_harness.py
class StructuredHarnessSystem(SystemUnderTest):
    """
    Invokes (stdin delivery, stdout capture — equivalent to Claude Code --print mode):
        structured-harness structured_configs/single_turn_qa.json -
                   --model <model_id>
                   --headless --autostart
        with full prompt passed via subprocess.communicate(input=prompt.encode())

    Model variants via --model flag:
      structured_harness (claude)  → --model sonnet
      structured_harness (gemini)  → --model gemini-2.0-flash
      structured_harness (gpt)     → --model gpt-4.1

    Single-codon assumption: single_turn_qa.json contains exactly one
    codon. The codon receives the stdin prompt and writes its answer to
    stdout. Output capture is identical to ClaudeSystem and GeminiSystem.
    """
```

**`single_turn_qa.json` design (assumed structure):**
```json
{
  "codons": [
    {
      "name": "answer_question",
      "prompt": "{{input}}"
    }
  ]
}
```
The `{{input}}` placeholder receives the stdin-piped prompt. The codon's response is captured from stdout. No output-file extraction needed.

### 6. Model-Aware Judge Rotation

The rotation strategy maps each SUT to a judge from a **different** underlying model provider:

| SUT | Underlying Model | Assigned Judge | Judge Model |
|---|---|---|---|
| `claude-code` | Anthropic | Google | gemini-2.0-flash |
| `gemini-cli` | Google | Anthropic | claude-opus-4-6 |
| `structured_harness (claude)` | Anthropic | OpenAI | gpt-4.1 |
| `structured_harness (gemini)` | Google | Anthropic | claude-opus-4-6 |
| `structured_harness (gpt)` | OpenAI | Anthropic | claude-opus-4-6 |

For the `fixed` strategy, a single judge (default: `gpt-4.1`) is used for all SUTs.

### 7. Metrics

| Metric | Category | Method |
|---|---|---|
| `exact_match` | Correctness | String normalization + equality |
| `soft_recall` | Correctness | Token-level F1 overlap. **Implementation:** Normalize string (lower, strip punctuation) -> split by whitespace -> compute F1. Do not use model-specific tokenizers. |
| `judge_score` | Correctness | EqualityJudge → 0/1 (CORRECT / INCORRECT) |
| `hallucination_rate` | Correctness | HallucinationJudge. **Implementation:** Use `nltk.sent_tokenize` to split response into sentences. Judge each sentence. Score = `(Contradicted + Unverified) / Total sentences`. |
| `latency_s` | Behavior | Wall-clock subprocess duration (includes CLI startup) |
| `verbosity` | Behavior | `len(prediction) / len(gold_answer)` |
| `refusal_rate` | Behavior | Regex: "I cannot answer", "I don't have access", etc. |
| `error_rate` | Reliability | `exit_code != 0` or `error_type is not None` |

**Primary metrics for hypothesis testing:** `judge_score` (success rate) and `hallucination_rate`.
All others are secondary/behavioral metrics reported in supplementary tables.

### 8. Human Evaluation Module

Validates automated judge quality. Operates as a separate set of CLI commands, independent from the main evaluation phases.

**Workflow:**
1. `glass export-human-eval <run_id> [--fraction 0.3]` — stratified export across all 7 domains, saved as `runs/{run_id}/human_eval/export_{date}.csv`. Default fraction: 0.3 (30 of 100 samples).
2. Annotators fill in the `human_label` column directly in the CSV (1 = CORRECT, 0 = INCORRECT).
3. `glass import-human-eval <run_id> <labels.csv>` — reads the completed file, validates all required rows are filled, attaches labels to the corresponding `EvalResult` objects.
4. `glass stats <run_id>` recomputes statistics including Cohen's Kappa (human vs. each automated judge) and appends to `statistics.json`.
5. Disagreement rows are flagged in `summary.md` for qualitative review in the error analysis section.

**Export CSV columns:**
```
sample_id, domain, question, gold_answer, system_name, prediction, automated_judge_score, human_label
```
`human_label` is blank on export. Annotators fill it in. All other columns are read-only reference.

**Target:** 20–40 samples (20–40% of 100 total), stratified so every domain has at least 2 annotated samples.

### 9. Error Analysis Module

Supports the "why" section of the paper. Operates on completed `EvalResult` objects.

**Capabilities:**
- Cross-tabulate error types (`timeout`, `refusal`, `malformed`, `crash`) by SUT and domain
- Identify samples where Structured Harness succeeds and baselines fail (and vice versa) — qualitative review candidates
- Verbosity vs. correctness scatter: does verbosity predict correctness?
- Hallucination breakdown: which domains see the highest HR?

Output: `error_analysis.md` appended to the run summary.

### 10. Prompt Templates (`glass/judges/prompts.py`)

To ensure reproducibility, judge prompts are stored as versioned constant strings in a dedicated Python module, not in YAML configs or scattered files.

```python
# glass/judges/prompts.py

AA_LCR_EQUALITY_V1 = """
Given the following question and gold answer, determine if the system's prediction is correct.
Question: {question}
Gold Answer: {gold_answer}
Prediction: {prediction}

Respond with exactly CORRECT or INCORRECT.
"""

NLI_SENTENCE_V1 = """
...
"""
```

### 11. Execution Flow

```
Phase 0 — Setup
  ├── Load + validate config.yaml via Pydantic schema
  ├── Generate run_id: {YYYYMMDD}_{HHMMSS}_{experiment.name}
  ├── Create runs/{run_id}/ directory tree
  ├── Write manifest.json (git hash, lib versions, config SHA256, timestamp)
  └── Log: "[GLASS] Run artifacts → runs/{run_id}/"

Phase 1 — Inference  (resumable via checkpoint.json)
  For each sample in dataset (ordered; stratified if subset):
    ├── Skip if checkpoint shows sample already completed for this SUT
    For each SUT:
      ├── Construct prompt via dataset adapter
      ├── subprocess.run(command, timeout=config.timeout_s)
      ├── Capture: stdout, stderr, exit_code, wall-clock latency
      ├── Classify error_type if applicable
      ├── Save RawOutput (with prompt and command) → runs/{run_id}/inference/{sut_name}/sample_{id}.json
      ├── Update rich TUI progress
      ├── Log: "[GLASS] Saved raw output → runs/{run_id}/inference/{sut_name}/sample_{id}.json"
      └── Update checkpoint.json

Phase 2 — Evaluation
  For each (sample, SUT) pair:
    ├── Load RawOutput from runs/{run_id}/inference/{sut_name}/sample_{id}.json
    ├── Run deterministic metrics (exact_match, soft_recall, verbosity, refusal, error)
    ├── If SUT errored (error_type is not None), set judge metrics to None and skip judges (AP-15)
    ├── strategy.assign_judge(sut_name) → judge
    ├── judge.evaluate_correctness(...) → judge_score + raw response
    ├── judge.evaluate_hallucination(...) → hallucination_rate + per-sentence labels
    ├── Save EvalResult → runs/{run_id}/evaluation/sample_{id}.json
    └── Append row to runs/{run_id}/results.csv

Phase 3 — Statistics
  ├── Load all EvalResults from runs/{run_id}/evaluation/
  ├── For each metric × SUT: compute mean, median, std
  ├── Bootstrap 95% CI (n=10,000 resamples per metric × SUT)
  ├── For primary hypothesis (judge_score, hallucination_rate):
  │     Wilcoxon signed-rank test, one-tailed (Structured Harness ≥ baseline)
  ├── For secondary metrics: two-tailed Wilcoxon
  ├── Effect size: rank-biserial correlation for all comparisons
  ├── Per-domain breakdown (exploratory; note sample-size caveat if N < 20)
  └── Write runs/{run_id}/statistics.json

Phase 4 — Reporting
  ├── Write runs/{run_id}/summary.md
  │     (aggregate table, per-domain table, statistical highlights, error type summary)
  └── Log: "[GLASS] Summary written → runs/{run_id}/summary.md"

Phase 5 — Human Eval  (separate CLI command, not automatic)
  ├── glass export-human-eval <run_id> [--fraction 0.3]
  ├── [external annotation step]
  ├── glass import-human-eval <run_id> labels.csv
  └── Compute Cohen's Kappa → appended to runs/{run_id}/statistics.json
```

### 11. Configuration Schema

```yaml
experiment:
  name: "runtime_comparison_v1"
  run_id: null              # Auto-generated if null: {date}_{time}_{name}
  seed: 42

dataset:
  name: "aa_lcr"
  samples: null             # null = all available; integer = stratified subset
  domains: null             # null = all 7; list = filter to specific domains

systems:
  - name: "claude-code"
    type: "claude"
    command: ["claude", "--print", "--output-format", "text"]  # Prompt via stdin
    timeout_s: 600                      # 100k-token contexts need generous timeout
    env: {}

  - name: "gemini-cli"
    type: "gemini"
    command: ["gemini", "-p"]           # -p triggers non-interactive; prompt via stdin
    timeout_s: 600
    env: {}

  - name: "codex-cli"
    type: "codex"
    command: ["codex", "exec", "--model", "gpt-5", "--ask-for-approval", "never"]
    timeout_s: 600
    env: {}

  - name: "structured_harness_claude"
    type: "structured_harness"
    model: "sonnet"                     # --model flag; same model family as claude-code
    harness_config: "single_turn_qa"    # → glass/systems/structured_configs/single_turn_qa.json
    timeout_s: 600
    env: {}

  - name: "structured_harness_gemini"
    type: "structured_harness"
    model: "gemini-2.0-flash"           # --model flag; same model family as gemini-cli
    harness_config: "single_turn_qa"
    timeout_s: 600
    env: {}

  - name: "structured_harness_gpt"
    type: "structured_harness"
    model: "gpt-4.1"
    harness_config: "single_turn_qa"
    timeout_s: 600
    env: {}

metrics:
  - exact_match
  - soft_recall
  - judge_score
  - hallucination_rate
  - latency_s
  - verbosity
  - refusal_rate
  - error_rate

judges:
  strategy: "rotation"      # "rotation" | "fixed"

  fixed:
    provider: "openai"
    model: "gpt-4.1"

  rotation:
    # model_family → judge config (no SUT may judge its own family)
    anthropic:
      provider: "google"
      model: "gemini-2.0-flash"
    google:
      provider: "anthropic"
      model: "claude-opus-4-6"
    openai:
      provider: "anthropic"
      model: "claude-opus-4-6"

  templates:
    correctness: "aa_lcr_equality"       # CORRECT / INCORRECT
    hallucination: "nli_sentence"        # SUPPORTED / CONTRADICTED / UNVERIFIED

statistics:
  bootstrap_resamples: 10000
  alpha: 0.05
  primary_test: "wilcoxon_one_tailed"    # For judge_score, hallucination_rate
  secondary_test: "wilcoxon_two_tailed"  # For all other metrics

output:
  runs_dir: "./runs"
  log_level: "INFO"
```

---

## Design Anti-Pattern Guide

> This section is required reading before implementing any component of GLASS. Each anti-pattern describes a real failure mode observed in evaluation harness codebases. Violations will silently corrupt results, making them unusable for publication.

---

### Category 1: Pipeline Architecture

**AP-1 — Mixing inference and evaluation in a single loop**

```python
# WRONG
for sample in dataset:
    output = sut.generate(sample.prompt)       # SUT call
    score = judge.evaluate(output, sample)     # Judge call in same loop
    results.append(score)
```
If the judge API fails at sample 80, all 80 SUT outputs are lost. The fix is two separate phases: Phase 1 writes every SUT output to disk before Phase 2 touches a single judge.

**AP-2 — Computing metrics without saving raw outputs**

Raw SUT outputs are the ground truth of the experiment. If you only save computed metric scores, you can never rerun a metric with a bug fix or add a new metric without re-invoking the SUT. Always save the full `RawOutput` (prompt, response, latency, stderr, exit code) before computing anything.

**AP-3 — Treating metric failure as a zero score**

```python
# WRONG
try:
    score = compute_f1(prediction, gold)
except Exception:
    score = 0.0   # silent corruption
```
A metric computation error and a genuinely wrong answer are different things. Metric failures must be recorded as `None` / `NaN` and surfaced in the run summary. A result row with a silent zero score is worse than a missing row.

**AP-4 — Running statistics inline during evaluation**

Do not compute means, CIs, or p-values while samples are still being processed. Statistics must be computed in a dedicated Phase 3 that reads the complete set of `EvalResult` objects. Partial-run statistics are meaningless and misleading.

---

### Category 2: SUT Invocation

**AP-5 — Using `shell=True` in subprocess**

```python
# WRONG
subprocess.run(f'claude --print "{prompt}"', shell=True)
```
Prompt text may contain quotes, backticks, `$()`, or newlines. With `shell=True`, these become shell injection vulnerabilities and will silently corrupt or abort the invocation. Always use a list of arguments and pass the prompt via stdin:

```python
# CORRECT
process = subprocess.Popen(
    ["claude", "--print", "--output-format", "text"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
stdout, stderr = process.communicate(input=prompt.encode("utf-8"), timeout=config.timeout_s)
```

**AP-6 — Passing large prompts as CLI arguments**

AA-LCR prompts are ~400KB. Linux's `ARG_MAX` is ~2MB total across all arguments and environment variables. Passing the prompt as a `-p "..."` argument is fragile and will fail silently or with a cryptic OS error on long contexts. Use stdin exclusively for prompt delivery.

**AP-7 — Ignoring stderr and exit codes**

A non-zero exit code with an empty stdout looks identical to an empty response if you only read stdout. Always capture stderr, always record exit code, and always classify the failure type (`timeout`, `api_error`, `refusal`, `malformed`, `crash`) before writing a `RawOutput`. Never let an error silently become an empty string that flows into metric computation.

**AP-8 — Not recording the exact command invoked**

The `RawOutput` schema must store the exact command list used (including all flags and the model version). Without this, you cannot reproduce a specific sample's result. Model versions change under aliases like `sonnet` — record the resolved version string when available.

**AP-9 — Interpreting latency as model speed**

Wall-clock subprocess time includes CLI startup, network round-trips, token generation, and CLI teardown. Do not label it "inference latency" or "TTFT". Call it `wall_clock_latency_s` and note in the paper what it includes. Comparing latency across SUTs with different CLI startup costs requires this caveat explicitly.

---

### Category 3: Judge Design

**AP-10 — Using a model to judge its own outputs**

A judge from the same model family as the SUT it is evaluating creates a well-documented self-preferencing bias. Gemini must not judge Gemini outputs; Claude must not judge Claude outputs. The rotation strategy in `judges/strategies/rotation.py` enforces this — do not bypass it.

**AP-11 — Not versioning judge prompts**

The `correctness` and `hallucination` prompt templates in `judges/` must be treated like source code: version-controlled, never edited in place between runs. A one-word change to a judge prompt can shift scores by several percentage points. If you change a judge prompt, start a new named template and update the config — never silently overwrite an existing template.

**AP-12 — Discarding raw judge responses**

The `EvalResult.judge_outputs` field stores the full text response from each judge call. This is mandatory for auditability — reviewers will ask "why was this sample marked INCORRECT?" and the only answer is the raw judge text. Never store only the parsed 0/1 score.

**AP-13 — Assuming judges are deterministic**

Even at `temperature=0`, LLM judge outputs can vary across API versions, infrastructure updates, and rate-limit retries. Do not re-run individual samples through a judge and assume the result is comparable to the original run. The evaluation phase must be treated as a one-shot operation; re-runs require a new run ID.

---

### Category 4: Metrics

**AP-14 — Exact match without normalization**

`"Paris" != "paris"` and `"42 million" != "42,000,000"`. Exact match must apply consistent normalization before comparison: lowercase, strip leading/trailing whitespace, normalize Unicode, remove trailing punctuation. Document the exact normalization steps — they are part of the metric definition.

**AP-15 — Computing hallucination rate on error outputs**

If a SUT times out or returns an empty string, its "hallucination rate" is 0/0 = undefined. Do not propagate this as `HR = 0.0` (which looks like a perfect score). Check `error_type` before running any LLM-based metric — if `error_type is not None`, set both `judge_score` and `hallucination_rate` to `None`, not zero.

**AP-16 — Conflating "wrong" with "refused/errored"**

A model that refuses to answer and a model that gives a confidently wrong answer are different failure modes and should be reported separately. The `refusal_rate` and `error_rate` metrics exist for this reason. Do not fold refusals into the correctness denominator without explicit disclosure.

---

### Category 5: Statistics

**AP-17 — Reporting means without confidence intervals**

With 100 samples and binary judge scores, a mean of 0.65 has a 95% CI of approximately ±0.09. Reporting "System A: 65%, System B: 60%" without CIs implies a precision that does not exist. Every reported aggregate metric must include bootstrap 95% CI. This is non-negotiable for publication.

**AP-18 — Using a paired t-test on binary judge scores**

Judge scores are Bernoulli (0 or 1). They are not normally distributed. The paired t-test is invalid for this data. Use Wilcoxon signed-rank test for paired comparisons. For the primary hypothesis, use the one-tailed variant (direction: Structured Harness ≥ baseline) as pre-specified — not chosen after seeing results.

**AP-19 — Claiming per-domain significance**

With ~14 samples per domain, no per-domain comparison will reach statistical significance. Do not run significance tests on domain subsets and report them as "significant at p<0.05." Report domain breakdowns as descriptive/exploratory only, with an explicit caveat in the paper. The power table in the Dataset Profile section documents this.

**AP-20 — Multiple comparisons without acknowledgment**

5 systems × 8 metrics = 40 pairwise comparisons. At α=0.05, you expect ~2 false positives by chance. The paper must acknowledge this. Primary claims are pre-specified (judge_score, hallucination_rate, Structured Harness vs. Claude Code and Gemini only). Secondary metrics are exploratory. Consider Bonferroni correction or Holm-Bonferroni for the primary comparison set if a reviewer requires it.

---

### Category 6: Reproducibility

**AP-21 — Not pinning model versions in SUTs**

`sonnet` resolves to different model versions over time. At run time, attempt to log the resolved model version (e.g., from the SUT's `--version` output or response metadata) into `manifest.json`. If the resolved version cannot be determined, log this explicitly. Never assume the alias is stable across runs.

**AP-22 — Editing config between phases**

The config used in Phase 1 (inference) must be identical to the config used in Phase 2 (evaluation). The frozen `config.yaml` in `runs/{run_id}/` is the canonical source of truth for a given run. Always load config from the run directory in Phase 2, not from the working directory. Mismatched configs (e.g., different judge settings) silently invalidate results.

**AP-23 — Not recording the random seed**

The stratified sampling in Phase 0 uses a seed. Without it, a "subset" run cannot be reproduced. The seed must be written to `manifest.json` and respected as a required config field, never optional.

---

### Category 7: Code Structure

**AP-24 — Hardcoding metric logic in the orchestrator**

```python
# WRONG — in pipeline/runner.py
if metric_name == "exact_match":
    score = prediction.lower() == gold.lower()
elif metric_name == "soft_recall":
    ...
```
This makes adding or changing metrics require modifying the orchestrator. Use the plugin registry: the orchestrator calls `registry.get(metric_name).compute(...)` and knows nothing about the metric's internals.

**AP-25 — Mutable default state in metric or judge classes**

Metric and judge classes must be stateless between samples. Do not accumulate running totals, caches, or history on the instance — if parallelism is ever added, this will cause data races. Aggregate statistics are computed in Phase 3 from the saved `EvalResult` objects, not inside metric classes.

**AP-26 — Catching broad exceptions silently**

```python
# WRONG
try:
    result = judge.evaluate(...)
except Exception:
    pass
```
This hides bugs in judge implementations, API errors, and malformed responses. Every caught exception must be logged with the full traceback and the affected sample ID. The result must be marked as `None` in the output, not silently dropped or zeroed.

**AP-27 — Assuming UTF-8 stdout from subprocesses**

CLI tools may output UTF-8, Latin-1, or include ANSI escape sequences. Always decode with `errors="replace"` and strip ANSI escape sequences before storing `RawOutput.output`. A corrupt encoding that gets stored and later compared against a gold answer will silently fail exact match and soft recall.

---

1. **`glass` Python package** — `pip install -e .` with `glass run`, `glass stats`, `glass export-human-eval`, `glass import-human-eval` CLI commands.
2. **Standard config files** — for single-turn QA (AA-LCR) in `systems/structured_configs/`.
3. **Config templates** — `aa_lcr_full.yaml`, `aa_lcr_subset.yaml`, `aa_lcr_ablation.yaml`.
4. **All plugin implementations** — metrics, judges, dataset adapters as described above.
5. **Statistical analysis module** — bootstrap CIs, Wilcoxon tests, effect sizes.
6. **Human eval tooling** — export/import workflow + Cohen's Kappa computation.
7. **Error analysis module** — for the qualitative "why" section of the paper.
8. **Report generator** — `summary.md`, `statistics.json`, optional LaTeX tables.
9. **README + replication guide** — step-by-step instructions for reproducing results from scratch.
10. **Independence & Disclosure** — README must include an explicit statement of independence, clarifying that while GLASS evaluates systems like the Structured Execution Harness, the harness and its findings are the product of independent research. It should include a standing invitation for third parties to run the harness and publish their results.

---

## Resolved Design Decisions

| # | Question | Decision |
|---|---|---|
| Q1 | Structured Harness output format | Treat as stdout (equivalent to Claude Code `--print` mode). Single-codon harness config, no output-file extraction. |
| Q2 | CLI stdin behavior | Confirmed: Claude uses `--print --output-format text`; Gemini uses `-p`. Both accept full prompt via stdin with the flag present. |
| Q3 | Evaluation phase parallelism | Sequential for simplicity and debuggability. |
| Q4 | Independence disclosure | Yes — README will include an explicit statement of independence and encourage third-party replication. |

---

## Implementation Roadmap

For an LLM or developer implementing GLASS, follow this build order to strictly adhere to the architecture:

0.  **Environment Setup:**
    *   Initialize a virtual environment (e.g., `venv`, `conda`).
    *   Install dependencies: `pip install pydantic pyyaml pandas numpy scipy datasets nltk click rich anthropic google-generativeai openai`.
    *   Ensure CLI tools (`claude`, `gemini`, `codex`) are installed and accessible in `$PATH`.
    *   Set API keys: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`.

1.  **Core Scaffolding:**
    *   `glass/config/schema.py` (Define Pydantic models first)
    *   `glass/systems/base.py` (Define SUT interface)
    *   `glass/judges/base.py` (Define Judge interface)
    *   `glass/metrics/base.py` (Define Metric interface)

2.  **Dataset & Templates:**
    *   `glass/datasets/aalcr.py` (Implement adapter)
    *   `glass/judges/prompts.py` (Add prompt constants)

3.  **System Implementations:**
    *   `glass/systems/claude.py`
    *   `glass/systems/gemini.py`
    *   `glass/systems/codex.py`
    *   `glass/systems/structured_harness.py`

4.  **Metric & Judge Logic:**
    *   `glass/metrics/*.py` (Implement deterministic metrics)
    *   `glass/judges/equality.py` & `hallucination.py`
    *   `glass/judges/strategies/rotation.py`

5.  **Orchestration (The "Gluer"):**
    *   `glass/cli.py` (CLI entry point)
    *   `glass/pipeline.py` (The inference -> evaluation -> stats loop)
    *   `glass/storage/` (Run saving/loading)

6.  **Statistics & Reporting:**
    *   `glass/statistics/` (Bootstrap & tests)
    *   `glass/reports/` (Markdown summary generation)

7.  **Verification:**
    *   Run `glass run configs/aa_lcr_subset.yaml` to verify end-to-end flow.
