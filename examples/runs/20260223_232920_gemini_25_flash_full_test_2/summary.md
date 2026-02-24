# GLASS Evaluation Summary

## Aggregate Metrics

> 95% bootstrap CI shown as [lower, upper]. See `statistics.json` for significance tests and effect sizes.

| System     | exact_match          | soft_recall          | soft_f1              | latency_s               | verbosity               | refusal_rate         | error_rate           | answer_length           | citation_presence    | confidence_score     | answer_completeness   | reasoning_depth      | response_consistency   | rouge_score_f1       | rouge_score_recall   | bert_score_f1        |
|:-----------|:---------------------|:---------------------|:---------------------|:------------------------|:------------------------|:---------------------|:---------------------|:------------------------|:---------------------|:---------------------|:----------------------|:---------------------|:-----------------------|:---------------------|:---------------------|:---------------------|
| gemini-cli | 0.010 [0.000, 0.031] | 0.648 [0.566, 0.728] | 0.161 [0.120, 0.205] | 43.392 [36.262, 51.779] | 61.817 [41.721, 86.689] | 0.000 [0.000, 0.000] | 0.040 [0.010, 0.080] | 74.229 [59.031, 91.771] | 0.000 [0.000, 0.000] | 0.516 [0.495, 0.542] | 1.000 [1.000, 1.000]  | 0.147 [0.099, 0.199] | 0.115 [0.086, 0.148]   | 0.150 [0.113, 0.191] | 0.631 [0.556, 0.704] | 0.696 [0.676, 0.717] |


## Per-Domain Breakdown (Exploratory)

> **Note (AP-19):** Per-domain samples are typically N≈14, which is insufficient for statistical significance claims. These numbers are descriptive only.

### Domain: Academia (N=5 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.953 |     0.183 |      28.791 |      19.029 |              0 |            0 |            80.4 |                   0 |                0.6 |                     1 |              0.24 |                  0.035 |            0.167 |                0.904 |           0.753 |


### Domain: Company_Documents (N=63 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |         0.016 |         0.656 |     0.171 |      44.698 |      59.779 |              0 |        0.032 |          80.738 |                   0 |                0.5 |                     1 |             0.148 |                  0.136 |            0.159 |                0.623 |           0.696 |


### Domain: Government_Consultations (N=11 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.749 |     0.236 |      59.668 |      61.246 |              0 |        0.182 |          45.667 |                   0 |              0.556 |                     1 |             0.122 |                  0.046 |            0.229 |                0.724 |           0.714 |


### Domain: Industry_Reports (N=8 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.375 |     0.019 |      39.636 |     171.804 |              0 |            0 |          92.125 |                   0 |                0.5 |                     1 |             0.225 |                  0.144 |            0.022 |                  0.5 |           0.623 |


### Domain: Legal (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.556 |     0.074 |      26.914 |      23.717 |              0 |            0 |          63.167 |                   0 |              0.583 |                     1 |             0.083 |                  0.078 |            0.063 |                0.522 |           0.679 |


### Domain: Marketing (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.557 |     0.156 |       34.45 |      20.214 |              0 |            0 |            35.5 |                   0 |                0.5 |                     1 |             0.083 |                  0.079 |             0.17 |                0.644 |           0.721 |


### Domain: Survey_Reports (N=1 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |             1 |     0.424 |      37.593 |       3.512 |              0 |            0 |              59 |                   0 |                0.5 |                     1 |                 0 |                  0.125 |            0.222 |                0.533 |           0.821 |

