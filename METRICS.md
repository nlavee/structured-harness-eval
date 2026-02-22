# GLASS Evaluation Metrics

This document defines every metric computed by the GLASS evaluation harness. Metrics are automatically computed during Phase 2 of the pipeline and aggregated in Phase 3.

All metrics are registered via `@register()` and subclass `BaseMetric`. They adhere to **AP-3** (return `None` on SUT errors, never silent `0.0`) and **AP-15** (return `None` when mathematically undefined).

---

## Metric Taxonomy

GLASS organises metrics into three categories:

| Category | Purpose | Statistical Role |
|:---|:---|:---|
| **Correctness** | Did the system answer correctly? | Primary hypothesis testing (`judge_score`, `hallucination_rate`) |
| **Behavioral** | How does the system *behave*? | Secondary / exploratory |
| **Operational** | Is the system reliable and fast? | Secondary / exploratory |

---

## Primary Metrics (Hypothesis Testing)

These metrics are used in the one-tailed Wilcoxon signed-rank test for the primary research claim.

### `judge_score`

| Property | Value |
|:---|:---|
| Category | Correctness |
| Range | `{0.0, 1.0}` or `None` |
| Requires Judge | Yes |
| AP Compliance | AP-3, AP-10, AP-12, AP-15 |

**Definition.** An LLM judge evaluates whether the prediction is semantically equivalent to the gold answer.

$$\text{judge\_score} = \begin{cases} 1.0 & \text{if judge responds CORRECT} \\ 0.0 & \text{if judge responds INCORRECT} \\ \text{None} & \text{if SUT error or judge failure} \end{cases}$$

**Cross-model comparison.** The primary metric for task success rate. When comparing System A (mean=0.72) vs System B (mean=0.65), the difference is meaningful only when bootstrap 95% CIs do not overlap. Judge rotation (AP-10) ensures no model judges its own outputs.

---

### `hallucination_rate`

| Property | Value |
|:---|:---|
| Category | Correctness |
| Range | `[0.0, 1.0]` or `None` |
| Requires Judge | Yes |
| AP Compliance | AP-3, AP-10, AP-15 |

**Definition.** The response is sentence-tokenised (`nltk.sent_tokenize`). Each sentence is classified by an LLM judge as SUPPORTED, CONTRADICTED, or UNVERIFIED against the source context.

$$\text{hallucination\_rate} = \frac{|\text{CONTRADICTED}| + |\text{UNVERIFIED}|}{|\text{total sentences}|}$$

**Cross-model comparison.** Lower is better. A system with HR=0.15 grounds claims more faithfully than one with HR=0.40. This metric captures whether structured execution harnesses reduce hallucination by enforcing explicit source grounding.

---

## Correctness Metrics (Deterministic)

### `exact_match`

| Property | Value |
|:---|:---|
| Category | Correctness |
| Range | `{0.0, 1.0}` or `None` |
| Requires Judge | No |
| AP Compliance | AP-3, AP-14 |

**Definition.** Strict string equality after normalisation (NFKD unicode, lowercase, article removal, punctuation stripping, whitespace collapse).

$$\text{exact\_match} = \mathbb{1}[\text{normalize}(\text{prediction}) = \text{normalize}(\text{gold})]$$

**Cross-model comparison.** Useful for factoid extraction tasks. Overly punitive for long-form answers — a correct answer with different phrasing scores 0.0. Always interpret alongside `judge_score`.

---

### `soft_recall`

| Property | Value |
|:---|:---|
| Category | Correctness |
| Range | `[0.0, 1.0]` or `None` |
| Requires Judge | No |
| AP Compliance | AP-3, AP-14, AP-15 |

**Definition.** Token-level recall using bag-of-words overlap after normalisation.

$$\text{soft\_recall} = \frac{|\text{pred\_tokens} \cap \text{gold\_tokens}|}{|\text{gold\_tokens}|}$$

Returns `None` if gold is empty.

**Cross-model comparison.** Measures what fraction of expected facts appeared in the output. A system with high recall but low precision is verbose but covers the ground truth.

---

### `soft_f1`

| Property | Value |
|:---|:---|
| Category | Correctness |
| Range | `[0.0, 1.0]` or `None` |
| Requires Judge | No |
| AP Compliance | AP-3, AP-14, AP-15 |

**Definition.** Token-level F1 (harmonic mean of precision and recall).

$$P = \frac{|\text{pred} \cap \text{gold}|}{|\text{pred}|}, \quad R = \frac{|\text{pred} \cap \text{gold}|}{|\text{gold}|}, \quad F_1 = \frac{2PR}{P + R}$$

**Cross-model comparison.** Balances precision and recall. A deterministic proxy for correctness that penalises both missing facts and extraneous content.

---

### `answer_completeness`

| Property | Value |
|:---|:---|
| Category | Correctness |
| Range | `[0.0, 1.0]` or `None` |
| Requires Judge | No |

**Definition.** For compound questions (multiple `?` marks), measures how many sub-questions are addressed.

$$\text{answer\_completeness} = \min\!\left(\frac{|\text{answer\_sentences}|}{\max(\text{question\_marks}, 1)},\; 1.0\right)$$

**Cross-model comparison.** Long-context QA questions often have multiple parts. Systems that address all parts score higher. Structured harnesses may perform better here due to explicit task decomposition.

---

## Behavioral Metrics

### `verbosity`

| Property | Value |
|:---|:---|
| Category | Behavioral |
| Range | `[0.0, ∞)` or `None` |
| Requires Judge | No |

**Definition.** Character-length ratio of prediction to gold answer.

$$\text{verbosity} = \frac{\text{len}(\text{prediction})}{\text{len}(\text{gold\_answer})}$$

Returns `None` if gold is empty.

**Cross-model comparison.** A ratio of 1.0 means the prediction matches gold length. Values >> 1.0 indicate excessive verbosity. Compare with `judge_score` to determine whether verbosity correlates with correctness.

---

### `answer_length`

| Property | Value |
|:---|:---|
| Category | Behavioral |
| Range | `[0, ∞)` or `None` |
| Requires Judge | No |

**Definition.** Word count of the prediction.

$$\text{answer\_length} = |\text{whitespace\_split}(\text{prediction})|$$

**Cross-model comparison.** A standalone measure, independent of gold answer length. Useful for detecting whether one runtime consistently produces longer outputs.

---

### `refusal_rate`

| Property | Value |
|:---|:---|
| Category | Behavioral |
| Range | `{0.0, 1.0}` or `None` |
| Requires Judge | No |
| AP Compliance | AP-16 |

**Definition.** Detects refusal via explicit `error_type="refusal"` or regex patterns ("I cannot answer", "I'm sorry", etc.). Returns `None` for non-refusal errors (crash, timeout) to avoid conflation (AP-16).

**Cross-model comparison.** A system with high refusal rate may be over-aligned for the task domain. Important for identifying safety-filter false positives in long-context reasoning.

---

### `citation_presence`

| Property | Value |
|:---|:---|
| Category | Behavioral |
| Range | `{0.0, 1.0}` or `None` |
| Requires Judge | No |

**Definition.** Binary detection of explicit citation syntax: `[doc N]`, `[N]` (1-3 digits), `Source: N`.

**Cross-model comparison.** Measures whether the system grounds claims with explicit references. RAG-style systems and structured harnesses may produce more citations.

---

### `confidence_score`

| Property | Value |
|:---|:---|
| Category | Behavioral |
| Range | `[0.0, 1.0]` or `None` |
| Requires Judge | No |

**Definition.** Ratio of confident vs. hedging language markers (based on Hyland's metadiscourse taxonomy).

$$\text{confidence\_score} = \frac{|\text{confident\_markers}|}{|\text{confident\_markers}| + |\text{hedging\_markers}|}$$

Returns 0.5 if no markers detected.

- **Confident markers**: "clearly", "certainly", "the answer is", "demonstrates", etc.
- **Hedging markers**: "might be", "perhaps", "I think", "seems to", etc.

**Cross-model comparison.** Captures calibration of epistemic language. A well-calibrated system should hedge when uncertain. Comparison with `judge_score` reveals over/under-confidence.

---

### `reasoning_depth`

| Property | Value |
|:---|:---|
| Category | Behavioral |
| Range | `[0.0, 1.0]` or `None` |
| Requires Judge | No |

**Definition.** Counts structural reasoning markers in three categories:

1. **Causal connectives**: "because", "therefore", "as a result"
2. **Enumeration**: numbered lists, bullet points, ordinal transitions
3. **Evidence grounding**: "according to", "based on", "the document states"

$$\text{reasoning\_depth} = \min\!\left(\frac{\text{total\_markers}}{10},\; 1.0\right)$$

**Cross-model comparison.** Structured execution harnesses may induce more explicit reasoning chains due to CoT scaffolding. This is directly relevant to the research hypothesis.

---

### `response_consistency`

| Property | Value |
|:---|:---|
| Category | Behavioral |
| Range | `[0.0, 1.0]` or `None` |
| Requires Judge | No |

**Definition.** Splits the response into two halves and compares entity mentions (numbers, proper nouns).

$$\text{response\_consistency} = \frac{|\text{entities\_first\_half} \cap \text{entities\_second\_half}|}{|\text{entities\_first\_half} \cup \text{entities\_second\_half}|}$$

Returns 1.0 if no entities detected (vacuously consistent). Returns `None` on empty output.

**Cross-model comparison.** Hallucinating systems often contradict themselves within a single response. Complements `hallucination_rate` without requiring a judge call.

---

## Operational Metrics

### `latency_s`

| Property | Value |
|:---|:---|
| Category | Operational |
| Range | `[0.0, ∞)` |
| Requires Judge | No |
| AP Compliance | AP-9 |

**Definition.** Wall-clock subprocess time in seconds. **Always returned** — even on SUT errors — because the measurement is always valid. A debug log is emitted when `error_type` is present.

**Cross-model comparison.** Includes CLI startup, network round-trips, and teardown (AP-9). Not a measure of pure inference speed.

---

### `error_rate`

| Property | Value |
|:---|:---|
| Category | Operational |
| Range | `{0.0, 1.0}` |
| Requires Judge | No |
| AP Compliance | AP-7, AP-15 |

**Definition.** Binary flag for SUT mechanical failure.

$$\text{error\_rate} = \mathbb{1}[\text{exit\_code} \neq 0 \;\lor\; \text{error\_type} \neq \text{None}]$$

**Always returns a value** (never `None`) — this metric is specifically about detecting errors.

**Cross-model comparison.** Crucial for system stability. If error_rate > 0, judge metrics are `None` for those samples (AP-15).

---

## Summary Table

| Metric | Category | Range | Judge? | Returns `None` on error? |
|:---|:---|:---|:---|:---|
| `judge_score` | Correctness | {0, 1} | ✓ | Yes |
| `hallucination_rate` | Correctness | [0, 1] | ✓ | Yes |
| `exact_match` | Correctness | {0, 1} | ✗ | Yes |
| `soft_recall` | Correctness | [0, 1] | ✗ | Yes |
| `soft_f1` | Correctness | [0, 1] | ✗ | Yes |
| `answer_completeness` | Correctness | [0, 1] | ✗ | Yes |
| `verbosity` | Behavioral | [0, ∞) | ✗ | Yes |
| `answer_length` | Behavioral | [0, ∞) | ✗ | Yes |
| `refusal_rate` | Behavioral | {0, 1} | ✗ | Yes (non-refusal errors) |
| `citation_presence` | Behavioral | {0, 1} | ✗ | Yes |
| `confidence_score` | Behavioral | [0, 1] | ✗ | Yes |
| `reasoning_depth` | Behavioral | [0, 1] | ✗ | Yes |
| `response_consistency` | Behavioral | [0, 1] | ✗ | Yes |
| `latency_s` | Operational | [0, ∞) | ✗ | **No** (always valid) |
| `error_rate` | Operational | {0, 1} | ✗ | **No** (always valid) |
