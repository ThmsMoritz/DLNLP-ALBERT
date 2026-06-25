# Extension: Low-Data Robustness

This folder contains the extension experiment. It compares BERT-base and ALBERT-base on the same tasks as the reproduction experiment, but with reduced training-data fractions.

Tasks:

- SST-2
- MRPC
- RTE
- MNLI (`validation_matched`)

Fractions:

- 10%
- 25%
- 50%

Default matrix config:

```text
configs/low_data_extension/low_data_matrix.yaml
```

Run the full extension:

```bash
python experiments/low_data_extension/run_low_data_extension.py
```

Useful subset commands:

```bash
python experiments/low_data_extension/run_low_data_extension.py --list-configs
python experiments/low_data_extension/run_low_data_extension.py --smoke-test
python experiments/low_data_extension/run_low_data_extension.py --only-task sst2 --only-fraction 0.1 --only-model albert-base-v2
```

Results are written to:

```text
results/low_data_results.csv
```
