# 5.56 Curriculum Ablation Analysis
Generated from 2 runs

| run | steps | bind_start→end | decay | stoch_div_max | gold_dist_red | mean_drift | prot |
|---|---|---|---|---|---|---|---|
| stochastic_ON | 6 | 0.4→0.1 | 0.3 | 4.0395 | 0.0 | 0.21666 | yes |
| ablation_ZERO | 6 | 0.4→0.1 | 0.3 | 0.0 | 0.0 | 0.66396 | yes |

## Key Observations (for Promotion Gate)
- **stochastic_ON**: strong stochastic breadth signal; clear scheduled decay; attractor protection was on throughout; 
- **ablation_ZERO**: clear scheduled decay; attractor protection was on throughout; 