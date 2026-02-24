# GLASS Evaluation Summary

## Aggregate Metrics

> 95% bootstrap CI shown as [lower, upper]. See `statistics.json` for significance tests and effect sizes.

| System     | exact_match          | soft_recall          | soft_f1              | latency_s               | verbosity               | refusal_rate         | error_rate           | answer_length           | citation_presence    | confidence_score     | answer_completeness   | reasoning_depth      | response_consistency   | rouge_score_f1       | rouge_score_recall   | bert_score_f1        |
|:-----------|:---------------------|:---------------------|:---------------------|:------------------------|:------------------------|:---------------------|:---------------------|:------------------------|:---------------------|:---------------------|:----------------------|:---------------------|:-----------------------|:---------------------|:---------------------|:---------------------|
| gemini-cli | 0.040 [0.010, 0.080] | 0.690 [0.612, 0.765] | 0.197 [0.149, 0.250] | 31.631 [29.651, 33.940] | 64.642 [44.993, 87.513] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 65.880 [55.750, 76.900] | 0.000 [0.000, 0.000] | 0.520 [0.505, 0.540] | 1.000 [1.000, 1.000]  | 0.145 [0.110, 0.185] | 0.095 [0.070, 0.123]   | 0.181 [0.136, 0.230] | 0.682 [0.608, 0.752] | 0.706 [0.685, 0.727] |


## Per-Domain Breakdown (Exploratory)

> **Note (AP-19):** Per-domain samples are typically N≈14, which is insufficient for statistical significance claims. These numbers are descriptive only.

### Domain: Academia (N=5 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.862 |     0.173 |      39.901 |      14.467 |              0 |            0 |            68.2 |                   0 |                0.6 |                     1 |              0.14 |                  0.077 |            0.166 |                0.851 |           0.752 |


### Domain: Company_Documents (N=63 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |         0.063 |         0.662 |     0.221 |      29.306 |       60.85 |              0 |            0 |          61.762 |                   0 |              0.508 |                     1 |              0.14 |                    0.1 |             0.19 |                0.631 |           0.708 |


### Domain: Government_Consultations (N=11 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.723 |     0.212 |      34.648 |      81.847 |              0 |            0 |          76.818 |                   0 |              0.545 |                     1 |             0.155 |                  0.089 |            0.207 |                0.808 |           0.708 |


### Domain: Industry_Reports (N=8 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |           0.5 |     0.017 |      38.816 |     150.332 |              0 |            0 |          89.625 |                   0 |                0.5 |                     1 |             0.225 |                  0.122 |            0.016 |                  0.5 |            0.62 |


### Domain: Legal (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.861 |     0.115 |      26.971 |       50.35 |              0 |            0 |            85.5 |                   0 |                0.5 |                     1 |             0.117 |                  0.091 |            0.102 |                0.867 |           0.688 |


### Domain: Marketing (N=6 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |         0.815 |     0.184 |      38.054 |      25.155 |              0 |            0 |          40.833 |                   0 |              0.583 |                     1 |             0.117 |                  0.022 |            0.296 |                  0.9 |           0.747 |


### Domain: Survey_Reports (N=1 samples)

| System     |   exact_match |   soft_recall |   soft_f1 |   latency_s |   verbosity |   refusal_rate |   error_rate |   answer_length |   citation_presence |   confidence_score |   answer_completeness |   reasoning_depth |   response_consistency |   rouge_score_f1 |   rouge_score_recall |   bert_score_f1 |
|:-----------|--------------:|--------------:|----------:|------------:|------------:|---------------:|-------------:|----------------:|--------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-----------------:|---------------------:|----------------:|
| gemini-cli |             0 |             1 |     0.622 |      35.446 |       2.276 |              0 |            0 |              36 |                   0 |                0.5 |                     1 |               0.1 |                  0.111 |             0.44 |                0.733 |           0.823 |

