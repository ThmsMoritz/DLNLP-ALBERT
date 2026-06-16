# DLNLP-ALBERT

## Overview

This project reproduces selected findings from **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** by Lan et al.

The project focuses on three claims:

1. ALBERT can achieve performance comparable to BERT with fewer parameters.
2. Cross-layer parameter sharing reduces model size with limited performance loss.
3. Factorized embeddings reduce parameter count without strongly hurting performance.

Because the original paper uses large-scale pretraining, this project performs smaller representative experiments on selected GLUE tasks.

## Datasets

### SST-2

- Task: binary sentiment classification
- Input: one sentence
- Metric: accuracy

### MRPC

- Task: paraphrase detection
- Input: sentence pair
- Metrics: accuracy and F1 score

### RTE

- Task: textual entailment
- Input: sentence pair
- Metric: accuracy

### MNLI

- Task: natural language inference
- Input: premise and hypothesis
- Metric: accuracy
- Default validation split: `validation_matched`

## Experiments

### Experiment 1: Reproduction

Compares pretrained BERT-base and ALBERT-base on selected GLUE tasks: SST-2, MRPC, RTE, and MNLI.

Models:

- `bert-base-uncased`
- `albert-base-v2`

Default configuration matrix:

```text
configs/reproduction/reproduction_matrix.yaml
```

Optional ALBERT size-scaling matrix:

```text
configs/reproduction/albert_size_scaling_matrix.yaml
```

This optional matrix compares `albert-base-v2`, `albert-large-v2`, and `albert-xlarge-v2` on SST-2, RTE, and MNLI. It is more expensive than the default baseline comparison.

Metrics:

- accuracy
- F1 score for MRPC
- parameter count
- training time
- optional GPU memory usage

Run:

```bash
python experiments/reproduction/run_reproduction.py
```

Useful subset runs:

```bash
# Inspect the expanded model × task grid.
python experiments/reproduction/run_reproduction.py --list-configs

# Verify the pipeline with tiny subsets before launching expensive runs.
python experiments/reproduction/run_reproduction.py --smoke-test

# Run only one task.
python experiments/reproduction/run_reproduction.py --only-task sst2

# Run only one model family.
python experiments/reproduction/run_reproduction.py --only-model albert
```

Output:

```text
results/reproduction_results.csv
```

### Experiment 2: Parameter Sharing Ablation

This experiment evaluates ALBERT's cross-layer parameter sharing mechanism by comparing multiple sharing strategies:

* No Sharing
* Shared Attention
* Shared FFN
* Full Sharing (standard ALBERT)

The implementation extends the standard Hugging Face ALBERT architecture by supporting attention-only and FFN-only sharing through selective parameter tying across ALBERT layer groups.

#### Extension: Layer-Position Parameter Sharing

To further investigate the role of parameter sharing within the transformer stack, an architectural extension was implemented:

* Lower-Half Sharing

  * Layers 1–6 share attention and FFN parameters
  * Layers 7–12 remain independent

* Upper-Half Sharing

  * Layers 1–6 remain independent
  * Layers 7–12 share attention and FFN parameters

This extension evaluates whether the position of parameter sharing within the network affects downstream task performance.

Experiments were conducted on two GLUE benchmark tasks:

* SST-2 (sentiment classification)
* MRPC (paraphrase detection)

#### Running the Experiment

#### Parameter Count Comparison

Compute parameter statistics without training:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py
```

#### SST-2 Fine-Tuning

Run all parameter-sharing configurations on SST-2:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train
```

Run only the layer-position sharing extension:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train \
  --config configs/parameter_sharing/lower_half_sharing.yaml \
  --config configs/parameter_sharing/upper_half_sharing.yaml
```

#### MRPC Fine-Tuning

Run the original parameter-sharing configurations on MRPC:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train \
  --config configs/parameter_sharing/no_sharing_mrpc.yaml \
  --config configs/parameter_sharing/shared_attention_mrpc.yaml \
  --config configs/parameter_sharing/shared_ffn_mrpc.yaml \
  --config configs/parameter_sharing/full_sharing_mrpc.yaml
```

Run the layer-position sharing extension on MRPC:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train \
  --config configs/parameter_sharing/lower_half_sharing_mrpc.yaml \
  --config configs/parameter_sharing/upper_half_sharing_mrpc.yaml
```

Outputs:

Results are appended to:

```text
results/parameter_sharing_results.csv
```

The generated results include parameter counts, evaluation metrics, training time, GPU memory usage, and task-specific metrics (e.g., F1 score for MRPC) for each configuration.


### Experiment 3: Factorized Embeddings

Compares standard embeddings and ALBERT-style factorized embeddings.

Configurations:

- standard embeddings
- factorized embeddings

Parameter-count run:

```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py
```

Optional training run:

```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py --train
```

Output:

```text
results/embedding_results.csv
```

## Extension: Low-Data Robustness

The extension compares BERT-base and ALBERT-base on the same GLUE tasks as the reproduction experiment, but with reduced training-data fractions.

Tasks:

- SST-2
- MRPC
- RTE
- MNLI (`validation_matched`)

Fractions:

- 10%
- 25%
- 50%

Default configuration matrix:

```text
configs/low_data_extension/low_data_matrix.yaml
```

Run:

```bash
python experiments/low_data_extension/run_low_data_extension.py
```

Useful subset runs:

```bash
# Inspect the expanded model × task × fraction grid.
python experiments/low_data_extension/run_low_data_extension.py --list-configs

# Verify the pipeline with tiny subsets before launching expensive runs.
python experiments/low_data_extension/run_low_data_extension.py --smoke-test

# Run only one task/fraction/model combination.
python experiments/low_data_extension/run_low_data_extension.py --only-task sst2 --only-fraction 0.1 --only-model albert-base-v2
```

Output:

```text
results/low_data_results.csv
```

## Project Structure

```text
.
├── configs/
│   ├── reproduction/
│   ├── parameter_sharing/
│   ├── factorized_embeddings/
│   └── low_data_extension/
├── experiments/
│   ├── reproduction/
│   ├── parameter_sharing/
│   ├── factorized_embeddings/
│   └── low_data_extension/
├── src/
│   ├── albert_project/
│   ├── parameter_count.py
│   └── plot_results.py
├── scripts/
├── results/
├── plots/
└── docs/
```

The `experiments/` folders are separated so each person can work on one experiment with minimal Git conflicts.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Plots

After experiments have produced result CSV files, run:

```bash
PYTHONPATH=src python src/plot_results.py
```

or:

```bash
bash scripts/generate_figures.sh
```

## Limitations

- Full BookCorpus/Wikipedia pretraining is not reproduced.
- The reproduction experiment uses pretrained Hugging Face checkpoints and fine-tunes task-specific classification heads.
- Ablation experiments may use randomly initialized controlled architectures.
- Attention-only and FFN-only sharing are implemented with module tying rather than separate pretrained checkpoints.
- Results may vary depending on hardware, random seed, and hyperparameters.

## Success Criteria

The project is successful if:

- ALBERT-base performs competitively with BERT-base while using fewer parameters.
- Parameter sharing clearly reduces model size.
- Factorized embeddings reduce parameter count.
- Trends are broadly consistent with the original ALBERT paper.
