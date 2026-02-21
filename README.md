# GLASS: Global Long-context Agent Scoring System

**GLASS** is a modular, open-source evaluation harness for comparing AI agent runtimes on long-context reasoning tasks. Inspired by the **Glass Frog**, it is designed for maximum transparency: every raw output, judge rationale, and intermediate artifact is preserved for rigorous auditability.

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

---

## Core Objectives

GLASS answers a precise research question: **Does a Structured Execution Harness achieve higher task success and lower hallucination rates than raw CLI runtimes while holding the underlying model constant?**

### Key Features
- **Transparent-Box Design:** Preserves 100% of raw SUT outputs and judge rationales.
- **Two-Phase Pipeline:** Decouples expensive model inference from evaluation/metrics.
- **Model-Aware Judge Rotation:** Eliminates self-preferencing bias by ensuring no model judges its own output.
- **Publication-Ready Stats:** Automated bootstrap confidence intervals and Wilcoxon signed-rank tests.

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

