# Stage B0C Sampler Audit

## Implemented Curriculum Schedule Format

Each phase has `start_step`, `end_step`, `train_n_pairs`, and optional `sampling_weights`; absent weights mean uniform sampling.

### B0C.1 phases

```json
[
  {
    "start_step": 0,
    "end_step": 5000,
    "train_n_pairs": [
      1,
      2
    ],
    "sampling_weights": "uniform"
  },
  {
    "start_step": 5000,
    "end_step": 10000,
    "train_n_pairs": [
      1,
      2,
      4
    ],
    "sampling_weights": "uniform"
  },
  {
    "start_step": 10000,
    "end_step": 30000,
    "train_n_pairs": [
      1,
      2,
      4,
      8
    ],
    "sampling_weights": "uniform"
  }
]
```

### B0C.2 phases

```json
[
  {
    "start_step": 0,
    "end_step": 5000,
    "train_n_pairs": [
      1,
      2
    ],
    "sampling_weights": "uniform"
  },
  {
    "start_step": 5000,
    "end_step": 10000,
    "train_n_pairs": [
      1,
      2,
      4
    ],
    "sampling_weights": "uniform"
  },
  {
    "start_step": 10000,
    "end_step": 20000,
    "train_n_pairs": [
      1,
      2,
      4,
      8
    ],
    "sampling_weights": "uniform"
  },
  {
    "start_step": 20000,
    "end_step": 35000,
    "train_n_pairs": [
      1,
      2,
      4,
      8,
      16
    ],
    "sampling_weights": "uniform"
  },
  {
    "start_step": 35000,
    "end_step": 50000,
    "train_n_pairs": [
      1,
      2,
      4,
      8,
      16,
      32,
      64
    ],
    "sampling_weights": "uniform"
  }
]
```

## Example Sampled Batches

```json
{
  "without_eval_first_20": [
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    4,
    1,
    2,
    4,
    2,
    4,
    1,
    2,
    1
  ],
  "with_eval_first_20": [
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    4,
    1,
    2,
    4,
    2,
    4,
    1,
    2,
    1
  ]
}
```

## Phase Boundary Proof

```json
{
  "1": [
    1,
    2
  ],
  "5000": [
    1,
    2
  ],
  "5001": [
    1,
    2,
    4
  ],
  "10000": [
    1,
    2,
    4
  ],
  "10001": [
    1,
    2,
    4,
    8
  ],
  "20000": [
    1,
    2,
    4,
    8
  ],
  "20001": [
    1,
    2,
    4,
    8,
    16
  ],
  "35000": [
    1,
    2,
    4,
    8,
    16
  ],
  "35001": [
    1,
    2,
    4,
    8,
    16,
    32,
    64
  ],
  "50000": [
    1,
    2,
    4,
    8,
    16,
    32,
    64
  ]
}
```

## Required Checks

- train/eval RNG are isolated: PASS
- eval_interval cannot affect train sample sequence: PASS
- labels are only non -100 at <ANS> positions: PASS; positions=[19]
- Transformer is autoregressive next-token LM: PASS; causal_delta=0.00000000
- no final-position classifier is used in B0C: PASS
- checkpoints are saved as latest, best_acc, and best_ce with optimizer and RNG state: PASS
- run config and commit hash are recorded: PASS; current_commit=e3668fa

## Conclusion

- PASS
