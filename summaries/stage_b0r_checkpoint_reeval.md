# Stage B0R Checkpoint Re-Evaluation

- eval_n_pairs: [1, 2, 4, 8]
- eval_seeds: [0, 1, 2, 3, 4]
- eval_samples_per_condition: 16384

## A2.2

- checkpoint: `/root/mamba_recall_experiment/runs/20260620_180244/checkpoints/stage_a2_n8_bestlr_to100k.pt`
- step: 6000
- N=8 mean acc: 0.9750
- N=8 mean CE: 0.0725
| eval_seed | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=8 CE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1250 | 0.1805 | 0.2255 | 0.9742 | 0.0763 |
| 1 | 0.1271 | 0.1791 | 0.2188 | 0.9734 | 0.0732 |
| 2 | 0.1300 | 0.1713 | 0.2151 | 0.9766 | 0.0730 |
| 3 | 0.1314 | 0.1761 | 0.2194 | 0.9744 | 0.0736 |
| 4 | 0.1281 | 0.1772 | 0.2197 | 0.9765 | 0.0663 |

## B0.1_seed0_best_acc

- checkpoint: `/root/mamba_recall_experiment/runs/20260620_180244/checkpoints/stage_b0_b01_n8_seed0_best_acc.pt`
- step: 17000
- N=8 mean acc: 0.3258
- N=8 mean CE: 1.5639
| eval_seed | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=8 CE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.5540 | 0.3993 | 0.3266 | 1.5617 |
| 1 | 1.0000 | 0.5676 | 0.4028 | 0.3306 | 1.5635 |
| 2 | 1.0000 | 0.5711 | 0.4016 | 0.3240 | 1.5663 |
| 3 | 1.0000 | 0.5639 | 0.3986 | 0.3217 | 1.5630 |
| 4 | 1.0000 | 0.5609 | 0.4027 | 0.3264 | 1.5652 |

## B0.1_seed1_best_acc

- checkpoint: `/root/mamba_recall_experiment/runs/20260620_180244/checkpoints/stage_b0_b01_n8_seed1_best_acc.pt`
- step: 13000
- N=8 mean acc: 0.3262
- N=8 mean CE: 1.5654
| eval_seed | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=8 CE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.0000 | 0.0417 | 0.3296 | 1.5643 |
| 1 | 1.0000 | 0.0001 | 0.0422 | 0.3273 | 1.5653 |
| 2 | 1.0000 | 0.0000 | 0.0397 | 0.3271 | 1.5667 |
| 3 | 1.0000 | 0.0001 | 0.0413 | 0.3226 | 1.5648 |
| 4 | 1.0000 | 0.0001 | 0.0406 | 0.3244 | 1.5660 |

## B0.1_seed2_best_acc

- checkpoint: `/root/mamba_recall_experiment/runs/20260620_180244/checkpoints/stage_b0_b01_n8_seed2_best_acc.pt`
- step: 20000
- N=8 mean acc: 0.3260
- N=8 mean CE: 1.5617
| eval_seed | N=1 acc | N=2 acc | N=4 acc | N=8 acc | N=8 CE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8518 | 0.5486 | 0.0243 | 0.3312 | 1.5592 |
| 1 | 0.8556 | 0.5527 | 0.0251 | 0.3318 | 1.5612 |
| 2 | 0.8518 | 0.5478 | 0.0261 | 0.3237 | 1.5636 |
| 3 | 0.8591 | 0.5483 | 0.0258 | 0.3194 | 1.5622 |
| 4 | 0.8547 | 0.5521 | 0.0275 | 0.3237 | 1.5624 |

## Interpretation

- A2.2 checkpoint remains high on fresh eval seeds, so the learned model is real; reproduction remains unresolved.
- B0 checkpoint N=8 mean accs: {'B0.1_seed0_best_acc': 0.32584228515625, 'B0.1_seed1_best_acc': 0.3261962890625, 'B0.1_seed2_best_acc': 0.3259521484375}
