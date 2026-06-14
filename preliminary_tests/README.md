# Preliminary Smoke Tests

This folder contains quick checks for verifying that the local environment and
training pipeline work before running the main experiments.

These files are intentionally kept outside `configs/` and `experiments/` so
smoke-test settings do not get confused with final experiment settings.

## Parameter Sharing Smoke Tests

Run from the project root:

```bash
python preliminary_tests/run_parameter_sharing_smoke.py
```

By default, the script trains a tiny full-sharing ALBERT-style model on a very
small SST-2 subset. To check the partial-sharing implementations, run:

```bash
python preliminary_tests/run_parameter_sharing_smoke.py --config preliminary_tests/parameter_sharing_smoke_attention.yaml
python preliminary_tests/run_parameter_sharing_smoke.py --config preliminary_tests/parameter_sharing_smoke_ffn.yaml
```

Smoke-test results are written to:

```text
results/preliminary_tests/parameter_sharing_smoke_results.csv
```

These preliminary result files can be deleted before final submission.
