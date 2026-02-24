# GLASS Evaluation Summary

## Aggregate Metrics

> 95% bootstrap CI shown as [lower, upper]. See `statistics.json` for significance tests and effect sizes.

| System     | exact_match          | soft_recall          | soft_f1              | latency_s               | verbosity                 | refusal_rate         | error_rate           | answer_length              | citation_presence    | confidence_score     | answer_completeness   | reasoning_depth      | response_consistency   | rouge_score_f1       | rouge_score_recall   | bert_score_f1        |
|:-----------|:---------------------|:---------------------|:---------------------|:------------------------|:--------------------------|:---------------------|:---------------------|:---------------------------|:---------------------|:---------------------|:----------------------|:---------------------|:-----------------------|:---------------------|:---------------------|:---------------------|
| gemini-cli | 0.000 [0.000, 0.000] | 0.741 [0.663, 0.813] | 0.093 [0.068, 0.120] | 53.998 [45.857, 63.023] | 122.967 [87.995, 162.906] | 0.000 [0.000, 0.000] | 0.020 [0.000, 0.050] | 121.367 [109.795, 133.000] | 0.000 [0.000, 0.000] | 0.592 [0.556, 0.633] | 1.000 [1.000, 1.000]  | 0.256 [0.206, 0.311] | 0.180 [0.151, 0.211]   | 0.083 [0.062, 0.106] | 0.741 [0.669, 0.807] | 0.653 [0.637, 0.669] |


## Per-Domain Breakdown (Exploratory)

> **Note (AP-19):** Per-domain samples are typically N≈14, which is insufficient for statistical significance claims. These numbers are descriptive only.

### Domain: Academia (N=5 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.942 |     0.148 |      63.858 |      26.437 |              0 |          0.2 |            91.5 |                   0 |                0.5 |                     1 |               0.2 |                  0.093 |            0.136 |                0.885 |           0.721 |


### Domain: Company_Documents (N=63 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |          0.72 |     0.084 |      53.882 |     132.219 |              0 |        0.016 |          129.21 |                   0 |              0.565 |                     1 |             0.271 |                  0.211 |            0.077 |                 0.73 |           0.644 |


### Domain: Government_Consultations (N=11 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.851 |     0.132 |      57.773 |      90.286 |              0 |            0 |         101.091 |                   0 |              0.636 |                     1 |             0.236 |                   0.13 |            0.117 |                0.824 |            0.67 |


### Domain: Industry_Reports (N=8 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.438 |     0.015 |      57.476 |     236.645 |              0 |            0 |         123.625 |                   0 |               0.75 |                     1 |             0.263 |                  0.139 |            0.017 |                0.438 |           0.601 |


### Domain: Legal (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.903 |     0.089 |      33.056 |      78.118 |              0 |            0 |         132.833 |                   0 |                0.5 |                     1 |             0.267 |                   0.14 |             0.08 |                0.909 |            0.66 |


### Domain: Marketing (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.815 |     0.091 |      58.554 |      64.995 |              0 |            0 |          96.667 |                   0 |               0.75 |                     1 |             0.183 |                  0.128 |            0.086 |                0.849 |           0.701 |


### Domain: Survey_Reports (N=1 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |             1 |     0.609 |      40.992 |       2.496 |              0 |            0 |              39 |                   0 |                0.5 |                     1 |               0.1 |                      0 |            0.431 |                0.733 |           0.824 |

