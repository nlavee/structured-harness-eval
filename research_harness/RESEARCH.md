# PhD-Level Research Formulation for GLASS Pipeline Runs

## 1. Objective
To systematically compare $N$ independent runs of the GLASS evaluation harness, extracting both quantitative rigor and qualitative nuances suitable for a PhD-level publication. The focus is on comparing the core hypothesis: **Structured execution harnesses achieve higher task success and lower hallucination rates than raw CLI models on long-horizon multi-document tasks.**

## 2. Quantitative Visualizations

To tease out nuances between models, we require high-density, rigorous visualizations:

### A. Distribution of Primary Metrics (Violin + Swarm Plots)
- **What:** Visualize the distribution of continuous scores (e.g., `hallucination_rate`, `soft_recall`) per system across all 100 samples.
- **Why:** Means hide the variance. We need to see if a system is consistently mediocre or bimodally perfect/failing.

### B. Bootstrap Confidence Interval Overlaps (Forest Plots)
- **What:** Plot the mean `judge_score` and `hallucination_rate` with their 95% CIs extracted directly from `statistics.json`.
- **Why:** The core statistical claim relies on CI separation and Wilcoxon signed-rank tests. This visualizes the statistical significance directly.

### C. Behavioral Radar Charts
- **What:** Normalize and plot secondary metrics (`verbosity`, `confidence_score`, `citation_presence`, `reasoning_depth`).
- **Why:** To profile the *persona* of each agent runtime. Does higher verbosity correlate with higher hallucination?

### D. Paired Sample Difference (Waterfall / Scatter)
- **What:** Plot $\Delta_{\text{metric}}$ for each sample (Structured Harness vs Baseline).
- **Why:** Evaluates paired improvements. Did the structured harness fix the baseline's failures, or are they failing on completely different samples?

## 3. Qualitative LLM Synthesis

Relying entirely on aggregate metrics misses the "why". We will utilize LLMs as hypothesis-generating research assistants.

### Synthesis Protocol
1. **Data Ingestion:** The LLM receives `results.csv` cross-tabs (samples where System A succeeded and System B failed), along with aggregate statistics and a description of the visualizations.
2. **Configurable Models:** To avoid model bias in the research itself, the synthesizer must allow switching between `claude-3-opus`, `gemini-1.5-pro` (or `2.0-flash`), and `gpt-4o`.
3. **Chain of Thought Logging:** The LLM's raw reasoning steps (how it notices patterns in errors) must be explicitly logged to disk (`research_insights/logs/`) *before* it outputs the final polished markdown.
4. **Scholarly Formatting:** The final output must read like a "Discussion & Error Analysis" section in an ACL/EMNLP paper, maintaining a strictly factual, objective, and hypothesis-driven tone.

## 4. Research Anti-Patterns (AP-RH)

To maintain PhD-level scientific rigor, this research harness explicitly defines and prevents the following anti-patterns (AP-RH):

### AP-RH1: Ignoring Sample Alignment (Unpaired Comparisons)
- **Anti-Pattern:** Taking the mean of Run A (e.g., $N=100$) and comparing it to Run B (e.g., $N=95$ due to SUT crashes) using independent statistical tests or raw means.
- **Correction:** The harness MUST perform an strict inner join on `sample_id` across all runs being compared. Paired statistical tests (Wilcoxon signed-rank, paired $\Delta$ plots) must *only* operate on the exact intersection of successful samples.

### AP-RH2: Aesthetic Over Scientific Visualizations (Dynamite Plots)
- **Anti-Pattern:** Using bar charts with standard error bars for non-normally distributed metrics (like `judge_score` or `hallucination_rate`). This hides bimodality, distribution skew, and obscures the actual $N$.
- **Correction:** Use Forest Plots for Bootstrap CIs, Violin + Swarm plots for distributions, and always overlay the number of samples ($N$) explicitly on the figure. 

### AP-RH3: Multi-Testing P-Hacking without Correction
- **Anti-Pattern:** When comparing $M > 2$ runs (e.g., Claude CLI vs Gemini CLI vs Structured Harness), running naive pairwise Wilcoxon tests and claiming significance at $\alpha = 0.05$ for the best result.
- **Correction:** The harness must automatically apply family-wise error rate corrections (e.g., Holm-Bonferroni correction) when emitting $p$-values across $M > 2$ configurations.

### AP-RH4: Ungrounded Qualitative LLM Hallucinations
- **Anti-Pattern:** Prompting the LLM Synthesizer to broadly "analyze the differences," resulting in the LLM inventing plausible-sounding but factually detached reasons for why System A beat System B based on its pre-defined biases.
- **Correction:** The LLM Synthesizer prompt must strictly require citation. The context must formulate exact paired divergences: "Sample ID 45: System A scored 1.0, System B scored 0.0. System A output: [...]. System B output: [...]". The LLM must synthesize insights *only* from the explicit divergence examples provided in context, cross-referenced with the quantitative data.

### AP-RH5: Re-computing Core Statistics Differently
- **Anti-Pattern:** The research harness recalculating global Bootstrap CIs differently (e.g., different resample count, different random seed) than the underlying run's `statistics.json`, leading to inconsistent reported numbers.
- **Correction:** The research harness must treat the `statistics.json` and `results.csv` from each individual run as the immutable, canonical source of truth for global metrics. It should only compute *new* statistics for strictly *cross-run* paired analyses (e.g., paired $\Delta$ CIs, rank-biserial effect sizes between runs).

### AP-RH6: Missing Automated Visual Interpretation
- **Anti-Pattern:** Generating high-quality visual plots but leaving them uninterpreted or only described superficially by the human researcher. Plots often contain nuanced statistical distributions that are easy to miss.
- **Correction:** For every generated plot, a Vision LLM should perform an automated visual analysis. The LLM must explicitly trace its reasoning in a `<thought>` block before outputting a PhD-level caption and interpretive findings for the visualization.

### AP-RH7: Hardcoded Research Prompts
- **Anti-Pattern:** Embedding the dense instructions for qualitative synthesis or visual interpretation directly inside the Python scripts. This prevents independent versioning and tracking of the methodology, and clutters orchestration code.
- **Correction:** All LLM prompts guiding research methodology must be explicitly extracted into a dedicated `prompts/` directory. Scripts must read these `.txt` files at runtime, ensuring complete separation of orchestration logic and research instruction text.
