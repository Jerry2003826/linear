# Stage B0E Sampler Audit

## Induction Multi-Query Format

- `single`: q_count = 1
- `all`: q_count = n_pairs
- `capped`: q_count = min(n_pairs, q_cap)
- B0E default: q_count_mode=capped, q_cap=8

## Curriculum Phases

### B0E.1

```json
[
  {
    "start_step": 0,
    "end_step": 5000,
    "train_n_pairs": [
      1,
      2
    ],
    "q_count_mode": "capped",
    "q_cap": 8,
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
    "q_count_mode": "capped",
    "q_cap": 8,
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
    "q_count_mode": "capped",
    "q_cap": 8,
    "sampling_weights": "uniform"
  }
]
```

### B0E.2

```json
[
  {
    "start_step": 0,
    "end_step": 5000,
    "train_n_pairs": [
      1,
      2
    ],
    "q_count_mode": "capped",
    "q_cap": 8,
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
    "q_count_mode": "capped",
    "q_cap": 8,
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
    "q_count_mode": "capped",
    "q_cap": 8,
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
    "q_count_mode": "capped",
    "q_cap": 8,
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
    "q_count_mode": "capped",
    "q_cap": 8,
    "sampling_weights": "uniform"
  }
]
```

## Decoded Examples

### n_pairs_4

```text
n_pairs=4
q_count_mode=capped
q_cap=8
q_count=4
record_keys=[122, 70, 83, 170]
query_keys=[70, 83, 122, 170]
full_tokens:
<BOS> <R> key_122 val_04 <R> key_070 val_02 <R> key_083 val_06 <R> key_170 val_06 <Q> key_070 val_02 <Q> key_083 val_06 <Q> key_122 val_04 <Q> key_170 val_06
labels:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 263, -100, -100, 267, -100, -100, 265, -100, -100, 267]
label_positions:
[14, 17, 20, 23]
```

### n_pairs_8

```text
n_pairs=8
q_count_mode=capped
q_cap=8
q_count=8
record_keys=[195, 149, 27, 93, 101, 219, 72, 52]
query_keys=[101, 149, 27, 195, 93, 52, 219, 72]
full_tokens:
<BOS> <R> key_195 val_04 <R> key_149 val_00 <R> key_027 val_02 <R> key_093 val_00 <R> key_101 val_00 <R> key_219 val_01 <R> key_072 val_05 <R> key_052 val_02 <Q> key_101 val_00 <Q> key_149 val_00 <Q> key_027 val_02 <Q> key_195 val_04 <Q> key_093 val_00 <Q> key_052 val_02 <Q> key_219 val_01 <Q> key_072 val_05
labels:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 261, -100, -100, 261, -100, -100, 263, -100, -100, 265, -100, -100, 261, -100, -100, 263, -100, -100, 262, -100, -100, 266]
label_positions:
[26, 29, 32, 35, 38, 41, 44, 47]
```

## q_count Counts And Label Positions

```json
{
  "q_counts": {
    "1": 1,
    "2": 2,
    "4": 4,
    "8": 8,
    "16": 8
  },
  "label_positions": {
    "1": [
      5
    ],
    "2": [
      8,
      11
    ],
    "4": [
      14,
      17,
      20,
      23
    ],
    "8": [
      26,
      29,
      32,
      35,
      38,
      41,
      44,
      47
    ],
    "16": [
      50,
      53,
      56,
      59,
      62,
      65,
      68,
      71
    ]
  }
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

- q_count test: PASS
- unique query test: PASS
- label alignment test: PASS
- answer-loss count test: PASS
- no final-position classifier test: PASS
- causal invariance test: PASS; causal_delta=0.00000000
- corrupted eval tests available: ['query_key_randomized', 'record_values_shuffled', 'records_removed', 'labels_shuffled']
- train/eval RNG remain isolated: PASS
- run config and commit hash are recorded: PASS; current_commit=6deb77b

## Conclusion

- PASS
