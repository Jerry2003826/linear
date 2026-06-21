# R3 Validation Split Summary

- seed: 42 (reproducible)
- sizes: train=3000, dev=1000, test=1000
- stratified jointly by (difficulty, conditions_bucket)
- disjoint & complete verified; same split used by all models

## Difficulty distribution (count / fraction)

| split | easy | medium | hard | adversarial |
|---|---|---|---|---|
| train | 872 (29.1%) | 1024 (34.1%) | 789 (26.3%) | 315 (10.5%) |
| dev | 290 (29.0%) | 343 (34.3%) | 262 (26.2%) | 105 (10.5%) |
| test | 290 (29.0%) | 343 (34.3%) | 262 (26.2%) | 105 (10.5%) |

## Conditions-per-query distribution

| split | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|
| train | 231 (7.7%) | 235 (7.8%) | 1795 (59.8%) | 739 (24.6%) |
| dev | 76 (7.6%) | 78 (7.8%) | 598 (59.8%) | 248 (24.8%) |
| test | 76 (7.6%) | 78 (7.8%) | 598 (59.8%) | 248 (24.8%) |
