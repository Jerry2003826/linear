# Stage B0D Sampler Audit

## Dense Multi-Query Format

- `single`: q_count = 1
- `all`: q_count = n_pairs
- `capped`: q_count = min(n_pairs, q_cap)
- B0D default: q_count_mode=capped, q_cap=8

## Curriculum Phases

### B0D.1

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

### B0D.2

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
<BOS> <K> key_122 <V> val_04 <K> key_070 <V> val_02 <K> key_083 <V> val_06 <K> key_170 <V> val_06 <Q> key_070 <ANS> val_02 <Q> key_083 <ANS> val_06 <Q> key_122 <ANS> val_04 <Q> key_170 <ANS> val_06
labels:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 265, -100, -100, -100, 269, -100, -100, -100, 267, -100, -100, -100, 269]
label_positions:
[19, 23, 27, 31]
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
<BOS> <K> key_195 <V> val_04 <K> key_149 <V> val_00 <K> key_027 <V> val_02 <K> key_093 <V> val_00 <K> key_101 <V> val_00 <K> key_219 <V> val_01 <K> key_072 <V> val_05 <K> key_052 <V> val_02 <Q> key_101 <ANS> val_00 <Q> key_149 <ANS> val_00 <Q> key_027 <ANS> val_02 <Q> key_195 <ANS> val_04 <Q> key_093 <ANS> val_00 <Q> key_052 <ANS> val_02 <Q> key_219 <ANS> val_01 <Q> key_072 <ANS> val_05
labels:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 263, -100, -100, -100, 263, -100, -100, -100, 265, -100, -100, -100, 267, -100, -100, -100, 263, -100, -100, -100, 265, -100, -100, -100, 264, -100, -100, -100, 268]
label_positions:
[35, 39, 43, 47, 51, 55, 59, 63]
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
      7
    ],
    "2": [
      11,
      15
    ],
    "4": [
      19,
      23,
      27,
      31
    ],
    "8": [
      35,
      39,
      43,
      47,
      51,
      55,
      59,
      63
    ],
    "16": [
      67,
      71,
      75,
      79,
      83,
      87,
      91,
      95
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
- run config and commit hash are recorded: PASS; current_commit=cea032d

## Conclusion

- PASS
