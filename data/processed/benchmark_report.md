# Final Benchmarking Report - Smart Scan EW Scheduler

## Summary Benchmark Table (Averaged across 10 Scenarios & Seeds)

| scheduler                     |       pd |       pfa |   avg_intercept_time |   spectrum_coverage |   total_retuning_cost |
|:------------------------------|---------:|----------:|---------------------:|--------------------:|----------------------:|
| LinUCB Bandit                 | 0.828996 | 0.0132689 |              24.9921 |            1        |               57.3667 |
| ML-Greedy Baseline            | 0.895397 | 0.0105377 |              18.0718 |            0.936667 |               41.5667 |
| Proposed Integrated Framework | 0.829249 | 0.0116115 |              23.3571 |            1        |               61.6    |
| Sequential Sweep              | 0.89     | 0.0140404 |              21.1224 |            1        |               99      |
| Uniform Random                | 0.928808 | 0.0114731 |              16.6274 |            1        |               87.6667 |

## Per-Scenario Performance Breakdown

### Scenario 1
| scheduler                     |   pd |        pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|-----:|-----------:|---------------------:|--------------------:|
| LinUCB Bandit                 |    1 | 0.00952381 |              4.66667 |            1        |
| ML-Greedy Baseline            |    1 | 0          |              2.83333 |            0.833333 |
| Proposed Integrated Framework |    1 | 0          |              4.66667 |            1        |
| Sequential Sweep              |    1 | 0.00833333 |              4.5     |            1        |
| Uniform Random                |    1 | 0.0121159  |              8.5     |            1        |

### Scenario 2
| scheduler                     |   pd |       pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|-----:|----------:|---------------------:|--------------------:|
| LinUCB Bandit                 |    1 | 0.0155672 |              4.92769 |                   1 |
| ML-Greedy Baseline            |    1 | 0.0118382 |              9.68287 |                   1 |
| Proposed Integrated Framework |    1 | 0.0156574 |              8.22973 |                   1 |
| Sequential Sweep              |    1 | 0.0137457 |             33.5862  |                   1 |
| Uniform Random                |    1 | 0.0108375 |             13.5661  |                   1 |

### Scenario 3
| scheduler                     |       pd |       pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|---------:|----------:|---------------------:|--------------------:|
| LinUCB Bandit                 | 0.333333 | 0.0134014 |              74.7667 |                   1 |
| ML-Greedy Baseline            | 0.666667 | 0.0139581 |              40.2    |                   1 |
| Proposed Integrated Framework | 0.333333 | 0.0134014 |              69.6667 |                   1 |
| Sequential Sweep              | 0        | 0.0133333 |             100      |                   1 |
| Uniform Random                | 1        | 0.0136412 |              17.175  |                   1 |

### Scenario 4
| scheduler                     |   pd |        pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|-----:|-----------:|---------------------:|--------------------:|
| LinUCB Bandit                 |    1 | 0.0156863  |             18.0789  |            1        |
| ML-Greedy Baseline            |    1 | 0.00883971 |              2.84717 |            0.966667 |
| Proposed Integrated Framework |    1 | 0.0157894  |             11.6203  |            1        |
| Sequential Sweep              |    1 | 0.0149813  |             23.9062  |            1        |
| Uniform Random                |    1 | 0.010989   |             16.4566  |            1        |

### Scenario 5
| scheduler                     |   pd |       pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|-----:|----------:|---------------------:|--------------------:|
| LinUCB Bandit                 |    1 | 0.017556  |              7.33333 |            1        |
| ML-Greedy Baseline            |    1 | 0.0168961 |             22       |            0.966667 |
| Proposed Integrated Framework |    1 | 0.0168803 |              7.33333 |            1        |
| Sequential Sweep              |    1 | 0.0117647 |              5       |            1        |
| Uniform Random                |    1 | 0.0154593 |              7.66667 |            1        |

### Scenario 6
| scheduler                     |   pd |        pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|-----:|-----------:|---------------------:|--------------------:|
| LinUCB Bandit                 |    1 | 0.0174649  |              16.3333 |                   1 |
| ML-Greedy Baseline            |    1 | 0.00813008 |              10.8333 |                   1 |
| Proposed Integrated Framework |    1 | 0.013092   |              11.8333 |                   1 |
| Sequential Sweep              |    1 | 0.0149813  |               4.5    |                   1 |
| Uniform Random                |    1 | 0.0111111  |               3.5    |                   1 |

### Scenario 7
| scheduler                     |       pd |       pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|---------:|----------:|---------------------:|--------------------:|
| LinUCB Bandit                 | 0.956631 | 0.0175439 |              6.66667 |            1        |
| ML-Greedy Baseline            | 0.953966 | 0.0185185 |             11.3333  |            0.933333 |
| Proposed Integrated Framework | 0.959154 | 0.0155123 |              6.66667 |            1        |
| Sequential Sweep              | 0.9      | 0.0148148 |              5.33333 |            1        |
| Uniform Random                | 0.954751 | 0.0113936 |             11       |            1        |

### Scenario 8
| scheduler                     |   pd |   pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|-----:|------:|---------------------:|--------------------:|
| LinUCB Bandit                 |    1 |  0    |              5.13333 |            1        |
| ML-Greedy Baseline            |    1 |  0    |              7.08333 |            0.766667 |
| Proposed Integrated Framework |    1 |  0    |              5.13333 |            1        |
| Sequential Sweep              |    1 |  0.02 |              4       |            1        |
| Uniform Random                |    1 |  0    |              5.73333 |            1        |

### Scenario 9
| scheduler                     |       pd |       pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|---------:|----------:|---------------------:|--------------------:|
| LinUCB Bandit                 | 0        | 0.0133333 |             100      |                   1 |
| ML-Greedy Baseline            | 0.333333 | 0.013367  |              72.5    |                   1 |
| Proposed Integrated Framework | 0        | 0.0133333 |             100      |                   1 |
| Sequential Sweep              | 1        | 0.013468  |              17.5    |                   1 |
| Uniform Random                | 0.333333 | 0.0134014 |              66.6667 |                   1 |

### Scenario 10
| scheduler                     |   pd |       pfa |   avg_intercept_time |   spectrum_coverage |
|:------------------------------|-----:|----------:|---------------------:|--------------------:|
| LinUCB Bandit                 |    1 | 0.012612  |             12.0143  |                 1   |
| ML-Greedy Baseline            |    1 | 0.013829  |              1.40441 |                 0.9 |
| Proposed Integrated Framework |    1 | 0.0124486 |              8.42113 |                 1   |
| Sequential Sweep              |    1 | 0.0149813 |             12.898   |                 1   |
| Uniform Random                |    1 | 0.0157824 |             16.0097  |                 1   |

