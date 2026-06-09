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

## Experiments

### Experiment 1: Reproduction

Compares pretrained BERT-base and ALBERT-base on SST-2 and MRPC.

Models:

- `bert-base-uncased`
- `albert-base-v2`

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

Output:

```text
results/reproduction_results.csv
```

### Experiment 2: Parameter Sharing Ablation

Compares ALBERT-style parameter sharing configurations.

Configurations:

- no sharing
- shared attention
- shared FFN
- full sharing

Important: Hugging Face ALBERT supports full/grouped sharing through `num_hidden_groups`, but does not directly support attention-only or FFN-only sharing. These configurations are marked with TODOs and skipped unless custom modeling code is added.

Parameter-count run:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --continue_on_todo
```

Optional training run:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train --continue_on_todo
```

Output:

```text
results/parameter_sharing_results.csv
```

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

The extension compares BERT-base and ALBERT-base when fine-tuned with different fractions of the training data.

Default task: SST-2.

Fractions:

- 10%
- 25%
- 50%
- 100%

Run:

```bash
python experiments/low_data_extension/run_low_data_extension.py
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
- The reproduction experiment uses pretrained Hugging Face checkpoints.
- Ablation experiments may use randomly initialized controlled architectures.
- Attention-only and FFN-only sharing require custom modeling code and are marked as TODO.
- Results may vary depending on hardware, random seed, and hyperparameters.

## Success Criteria

The project is successful if:

- ALBERT-base performs competitively with BERT-base while using fewer parameters.
- Parameter sharing clearly reduces model size.
- Factorized embeddings reduce parameter count.
- Trends are broadly consistent with the original ALBERT paper.
