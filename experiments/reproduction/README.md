# Experiment 1: Reproduction

Owner: Person responsible for the reproduction experiment.

This folder contains the script for the pretrained BERT-base vs ALBERT-base comparison.
The default run expands `configs/reproduction/reproduction_matrix.yaml` into a model × task grid:

- Models: `bert-base-uncased`, `albert-base-v2`
- Tasks: SST-2, MRPC, RTE, MNLI matched validation

An optional size-scaling config is also provided at `configs/reproduction/albert_size_scaling_matrix.yaml` for ALBERT-base/large/xlarge on SST-2, RTE, and MNLI.

Run all default configurations:

```bash
python experiments/reproduction/run_reproduction.py
```

Inspect the expanded run matrix:

```bash
python experiments/reproduction/run_reproduction.py --list-configs
```

Run a quick pipeline check without producing final-report results:

```bash
python experiments/reproduction/run_reproduction.py --smoke-test
```

Run a subset:

```bash
python experiments/reproduction/run_reproduction.py --only-task sst2
python experiments/reproduction/run_reproduction.py --only-model albert
python experiments/reproduction/run_reproduction.py --config configs/reproduction/albert_base_sst2.yaml
python experiments/reproduction/run_reproduction.py --config configs/reproduction/albert_size_scaling_matrix.yaml --only-task sst2
```

Results are appended to:

```text
results/reproduction_results.csv
```

Important: `--smoke-test` writes tiny-subset results and should only be used to verify the pipeline.
Final reported results should come from normal runs without `max_train_samples` or `max_eval_samples`.
