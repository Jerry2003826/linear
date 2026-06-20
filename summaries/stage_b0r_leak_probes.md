# Stage B0R Leak Probes

| probe | acc | CE |
| --- | ---: | ---: |
| normal | 0.9740 | 0.0821 |
| query_randomized | 0.1819 | 5.1854 |
| values_shuffled | 0.2383 | 7.6654 |
| records_removed | 0.1267 | 5.6787 |
| labels_shuffled | 0.1322 | 8.8922 |

## Interpretation

- PASS: normal eval is high and corrupted probes drop, supporting real association behavior.
