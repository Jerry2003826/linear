# Stage B0R Data And Label Fingerprint Audit

## A2 train

- bad_count: 0
- first bad examples: []
- seq_lens: [36]
- value_counts: [1326, 1298, 1164, 1237, 1231, 1252, 1271, 1221]
- unique_hashes: 10000 / 10000
- repeats: 0
- first query positions: [34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]

## A2 eval

- bad_count: 0
- first bad examples: []
- seq_lens: [36]
- value_counts: [1266, 1275, 1278, 1210, 1280, 1236, 1219, 1236]
- unique_hashes: 10000 / 10000
- repeats: 0
- first query positions: [34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]

## B0 train

- bad_count: 0
- first bad examples: []
- seq_lens: [36]
- value_counts: [1326, 1298, 1164, 1237, 1231, 1252, 1271, 1221]
- unique_hashes: 10000 / 10000
- repeats: 0
- first query positions: [34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]

## B0 eval

- bad_count: 0
- first bad examples: []
- seq_lens: [36]
- value_counts: [1266, 1275, 1278, 1210, 1280, 1236, 1219, 1236]
- unique_hashes: 10000 / 10000
- repeats: 0
- first query positions: [34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]

## Checks

- unique keys per sample: PASS
- query key comes from records: PASS
- answer equals queried value: PASS
- non-ignore labels == q_count: PASS
- <ANS> label is correct value token: PASS
- N=8 sequence length == 36: PASS
- token id ranges are non-overlapping: PASS
- value distribution is roughly uniform: PASS
- query position is stable by format: PASS
- no abnormal duplicate hashes: PASS
- train/eval are not identical due different seeds: PASS
- A2/B0 configs match: PASS

## Conclusion

- PASS
