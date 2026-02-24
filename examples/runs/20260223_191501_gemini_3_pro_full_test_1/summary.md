# GLASS Evaluation Summary

## Aggregate Metrics

> 95% bootstrap CI shown as [lower, upper]. See `statistics.json` for significance tests and effect sizes.

| System     | exact_match          | soft_recall          | soft_f1              | latency_s               | verbosity                 | refusal_rate         | error_rate           | answer_length             | citation_presence    | confidence_score     | answer_completeness   | reasoning_depth      | response_consistency   | rouge_score_f1       | rouge_score_recall   | bert_score_f1        |
|:-----------|:---------------------|:---------------------|:---------------------|:------------------------|:--------------------------|:---------------------|:---------------------|:--------------------------|:---------------------|:---------------------|:----------------------|:---------------------|:-----------------------|:---------------------|:---------------------|:---------------------|
| gemini-cli | 0.000 [0.000, 0.000] | 0.755 [0.668, 0.836] | 0.106 [0.074, 0.145] | 46.684 [41.872, 52.113] | 119.383 [78.213, 167.722] | 0.000 [0.000, 0.000] | 0.270 [0.180, 0.360] | 106.452 [94.164, 119.987] | 0.000 [0.000, 0.000] | 0.575 [0.534, 0.623] | 0.993 [0.979, 1.000]  | 0.200 [0.156, 0.251] | 0.135 [0.107, 0.166]   | 0.099 [0.068, 0.135] | 0.756 [0.672, 0.833] | 0.658 [0.639, 0.678] |


## Per-Domain Breakdown (Exploratory)

> **Note (AP-19):** Per-domain samples are typically N≈14, which is insufficient for statistical significance claims. These numbers are descriptive only.

### Domain: Academia (N=5 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.942 |     0.123 |      37.502 |      25.353 |              0 |          0.2 |           94.25 |                   0 |               0.75 |                     1 |               0.2 |                  0.101 |            0.111 |                0.885 |           0.728 |


### Domain: Company_Documents (N=63 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |          0.73 |     0.093 |      48.164 |     117.754 |              0 |         0.27 |         108.587 |                   0 |              0.576 |                 0.989 |             0.183 |                  0.146 |            0.089 |                0.742 |           0.651 |


### Domain: Government_Consultations (N=11 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.962 |     0.243 |      42.934 |     124.416 |              0 |        0.364 |             104 |                   0 |              0.571 |                     1 |             0.129 |                  0.063 |            0.234 |                0.957 |           0.686 |


### Domain: Industry_Reports (N=8 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |           0.5 |     0.012 |      54.997 |     250.354 |              0 |        0.125 |         127.286 |                   0 |                0.5 |                     1 |             0.314 |                  0.171 |            0.014 |                0.571 |           0.602 |


### Domain: Legal (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.833 |      0.03 |      32.015 |     156.812 |              0 |        0.667 |           107.5 |                   0 |               0.25 |                     1 |              0.55 |                  0.211 |            0.027 |                0.833 |           0.586 |


### Domain: Marketing (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.815 |     0.132 |      51.417 |      42.499 |              0 |            0 |              81 |                   0 |              0.583 |                     1 |             0.183 |                  0.068 |            0.113 |                0.765 |           0.705 |


### Domain: Survey_Reports (N=1 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |             1 |     0.341 |      33.757 |        4.85 |              0 |            0 |              79 |                   0 |                  1 |                     1 |               0.1 |                   0.25 |            0.172 |                0.533 |           0.799 |

