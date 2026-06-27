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



# Experiment 3 — Factorized Embeddings

Reproduction and extension of ALBERT's factorized embedding parameterization: factorizing
the token-embedding matrix (V x H into V x E + E x H) reduces parameters without strongly
hurting performance. Reproduced on SST-2 and MRPC, with an embedding-size sweep as extension.

## Notebooks

- `exp3_factorized_embeddings.ipynb` — main notebook: parameter analysis, SST-2 reproduction, and the embedding-size sweep extension.
- `mrpc.ipynb` — companion notebook: factorized-embedding reproduction on MRPC.

## Method

Standard and factorized configurations are identical except for `embedding_size`
(768 vs 128), isolating factorization as the only changed variable. Models are randomly
initialized (`from_config`) and fine-tuned for 3 epochs. Results are appended to
`results/embedding_results.csv`.

## 1. Reproduction on SST-2

Standard (E=768) vs factorized (E=128) embeddings on SST-2.

Parameter-count only (no training, fast):
```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py
```

Full fine-tuning (uses the default SST-2 configs):
```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py --train
```

Result: factorization cuts embedding parameters about 6x (23.4M to 3.9M) and total
parameters about 63% (31.7M to 11.7M), at about 1 point of accuracy cost
(standard 82.5% vs factorized 81.3%).

## 2. Reproduction on MRPC

Same comparison on MRPC, a sentence-pair paraphrase task scored with accuracy and F1.

```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py --train \
  --config configs/factorized_embeddings/standard_embeddings_mrpc.yaml \
  --config configs/factorized_embeddings/factorized_embeddings_mrpc.yaml
```

Result: same parameter reduction. Both models converge to the majority-class baseline
(about 68% accuracy / 81% F1), expected for randomly-initialized models on MRPC's small
training set (about 3,700 pairs). Both behave identically, so the comparison stays valid.

## 3. Extension — embedding-size sweep

Sweeps the embedding size E over {64, 128, 768} on SST-2 (20% of the data, via
`train_fraction: 0.2`) to map the accuracy/parameter trade-off and ask whether ALBERT's
choice of a small E is justified.

```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py --train \
  --config configs/factorized_embeddings/sweep_e64_sst2.yaml \
  --config configs/factorized_embeddings/sweep_e128_sst2.yaml \
  --config configs/factorized_embeddings/sweep_e768_sst2.yaml
```

Result: accuracy rises only about 4.6 points (73.2% to 77.8%) as E grows 12x (64 to 768),
while total parameters more than triple — strongly diminishing returns that support
ALBERT's choice of a small embedding size.

## Configs

In `configs/factorized_embeddings/`:
- `standard_embeddings.yaml`, `factorized_embeddings.yaml` — SST-2 (E=768 vs E=128)
- `standard_embeddings_mrpc.yaml`, `factorized_embeddings_mrpc.yaml` — MRPC
- `sweep_e64_sst2.yaml`, `sweep_e128_sst2.yaml`, `sweep_e768_sst2.yaml` — extension sweep

## Outputs

- `results/embedding_results.csv` — all runs (parameters, accuracy, F1, timing).
- `plots/extension_accuracy_vs_E.png` — accuracy vs embedding size figure.

## Notes

Models are randomly initialized rather than pretrained, so absolute accuracy is modest;
the relative comparisons (standard vs factorized, and accuracy vs E) are the valid results.
Training requires `accelerate` and a GPU is recommended. Run commands from the repo root.


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
