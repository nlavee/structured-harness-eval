# GLASS: Global Long-context Agent Scoring System

**GLASS** is a modular, open-source evaluation harness for comparing AI agent runtimes on long-context reasoning tasks. Inspired by the **Glass Frog**, it is designed for maximum transparency: every raw output, judge rationale, and intermediate artifact is preserved for rigorous auditability.

---

## Examples

To see GLASS in action, browse the **[examples/](./examples/)** directory.
- **[Example Run Artifacts](./examples/runs/)**: Raw outputs, results, and rigorous statistics.
- **[Example Research Insights](./examples/research_insights/)**: Distribution-aware visualizations and synthesized paper-ready Discussion sections.

---

## Quick Start

**1. Install GLASS:**

```bash
git clone https://github.com/nlavee/structured-harness-eval
cd glass
pip install -e .
```

**2. Download the AA-LCR dataset:**

```bash
python scripts/download_aalcr.py
```

**3. Run a full evaluation:**

```bash
glass run configs/aa_lcr_full.yaml
```

**4. Compute statistics on a completed run:**

```bash
glass stats runs/<run_id>
```

**5. Re-evaluate a previous run with new metrics:**

```bash
glass run configs/new_metrics.yaml --re-evaluate <run_id>
```

**6. Synthesize scientific insights across multiple runs:**

```bash
python3 research_harness/cli.py --runs <run_id_1> <run_id_2> --provider openai --model gpt-4o-mini
```

---

## Task Runner

GLASS includes a `robo` task runner to simplify common operations.

**Install Robo:**
```bash
# Install to ./bin
curl -sfL https://github.com/tj/robo/releases/download/v0.5.4/robo_linux_amd64 -o bin/robo && chmod +x bin/robo

# Add to PATH (optional)
export PATH=$PATH:$(pwd)/bin
```

**Run an evaluation:**
```bash
bin/robo eval:run configs/aa_lcr_full.yaml
```

**Re-evaluate an existing inference run:**
```bash
bin/robo eval:re-eval configs/new_metrics.yaml <run_id>
```

**Launch the research harness:**
```bash
bin/robo research:harness --runs <run_id_1> <run_id_2>
```

**View all available tasks:**
```bash
bin/robo help
```

---

## Core Objectives

GLASS answers a precise research question: **Does a Structured Execution Harness achieve higher task success and lower hallucination rates than raw CLI runtimes while holding the underlying model constant?**

### Key Features
- **Transparent-Box Design:** Preserves 100% of raw SUT outputs and judge rationales, including full prompts and commands.
- **Two-Phase Pipeline:** Decouples expensive model inference from evaluation/metrics, featuring a rich Terminal User Interface (TUI) with real-time progress and dual logging.
- **Robust Error Handling:** Correctly handles timeouts, system crashes, and API failures by explicitly marking metrics as missing instead of corrupting downstream statistics.
- **Model-Aware Judge Rotation:** Eliminates self-preferencing bias by ensuring no model judges its own output.
- **Publication-Ready Stats:** Automated bootstrap confidence intervals and Wilcoxon signed-rank tests.
- **Research Harness (`research_harness/`)**: 
  - Cross-run aggregation enforcing scientific Anti-Patterns (AP-RH1 to AP-RH7).
  - Pydantic-validated data pipelines extracting Domain-level shifts.
  - Generates distribution-aware visualizations (Violin, Swarm, Forest CIs, Domain Heatmaps).
  - Configurable LLM Qualitative Synthesizer for paper-ready Discussion & Error Analysis grounded in Paired Divergences.
  - **Multimodal Vision Interpretation**: Automated analysis of visualizations using batched multimodal LLM calls for improved cross-view context and efficiency.
  - **Centralized Naming Utility**: Consistent file naming and metadata tracking via `naming.py`.

---

## Systems & Dataset

GLASS benchmark defaults to the [`ArtificialAnalysis/AA-LCR`](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) dataset (100 multi-document questions) and evaluates:
- **Claude Code** vs. **Structured Execution Harness (Claude)**
- **Gemini CLI** vs. **Structured Execution Harness (Gemini)**
- **Codex CLI (GPT 5)** vs. **Structured Execution Harness (GPT 5)**

---

## Independence & Disclosure

GLASS is an **independent evaluation framework**. While it evaluates systems like the Structured Execution Harness, the harness and its findings are the product of independent research. We enforce objectivity through open artifacts, pre-specified hypotheses, and mandatory human evaluation spot-checks.

---

## Full Documentation

- **Research & Architecture:** 👉 **[PLAN.md](./PLAN.md)**
- **Developer & Usage Guide:** 👉 **[GUIDE.md](./GUIDE.md)**

---

## Citation

```bibtex
@misc{glass2026,
  title  = {GLASS: Global Long-context Agent Scoring System},
  author = {Vu, N.},
  year   = {2026},
  url    = {https://github.com/nlavee/structured-harness-eval}
}
```

