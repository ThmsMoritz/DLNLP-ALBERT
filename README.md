# DLNLP-ALBERT

## Overview

This project reproduces selected findings from the paper **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** by Lan et al.

The paper proposes ALBERT as a parameter-efficient version of BERT. Its main ideas are:

- factorized embedding parameterization
- cross-layer parameter sharing
- Sentence Order Prediction (SOP)

In this project, we focus on the architectural efficiency claims of ALBERT and compare them against BERT in a smaller, reproducible setup.

## Objective

The goal is to validate three main claims:

1. ALBERT achieves performance comparable to BERT while using fewer parameters.
2. Cross-layer parameter sharing reduces model size with limited performance loss.
3. Factorized embeddings reduce parameter count without strongly affecting performance.

## Datasets

We use two GLUE-style classification tasks.

### SST-2

Binary sentiment classification.

- Input: one sentence
- Output: positive or negative
- Metric: accuracy

### MRPC

Paraphrase detection.

- Input: sentence pair
- Output: paraphrase or not paraphrase
- Metrics: accuracy and F1 score

## Experiment 1: Reproduction

### Goal

Compare BERT-base and ALBERT-base on downstream task performance and parameter efficiency.

### Models

- `bert-base-uncased`
- `albert-base-v2`

### Tasks

- SST-2
- MRPC

### Metrics

- SST-2 accuracy
- MRPC accuracy
- MRPC F1 score
- parameter count
- training time

### Expected Outcome

ALBERT-base should achieve performance close to BERT-base while using significantly fewer parameters.

## Experiment 2: Parameter Sharing Ablation

### Goal

Analyze how cross-layer parameter sharing affects model size and performance.

### Configurations

- no sharing
- shared attention
- shared FFN
- full sharing

### Tasks

- SST-2
- MRPC

### Metrics

- accuracy
- F1 score
- parameter count
- training time
- optional GPU memory usage

### Expected Outcome

Full parameter sharing should produce the largest reduction in parameters, with some possible performance loss.

## Experiment 3: Factorized Embeddings

### Goal

Analyze whether factorized embeddings reduce parameter count while preserving performance.

### Configurations

- standard embeddings
- factorized embeddings

### Tasks

- SST-2
- MRPC

### Metrics

- accuracy
- F1 score
- embedding parameter count
- total parameter count

### Expected Outcome

Factorized embeddings should significantly reduce parameter count with only minor performance changes.

## Results

### Reproduction Results

| Model | SST-2 Accuracy | MRPC Accuracy | MRPC F1 | Parameters | Training Time |
|---|---:|---:|---:|---:|---:|
| BERT-base | TBD | TBD | TBD | TBD | TBD |
| ALBERT-base | TBD | TBD | TBD | TBD | TBD |

### Parameter Sharing Results

| Configuration | SST-2 Accuracy | MRPC Accuracy | MRPC F1 | Parameters | Memory Usage |
|---|---:|---:|---:|---:|---:|
| No Sharing | TBD | TBD | TBD | TBD | TBD |
| Shared Attention | TBD | TBD | TBD | TBD | TBD |
| Shared FFN | TBD | TBD | TBD | TBD | TBD |
| Full Sharing | TBD | TBD | TBD | TBD | TBD |

### Embedding Results

| Configuration | SST-2 Accuracy | MRPC Accuracy | MRPC F1 | Parameters |
|---|---:|---:|---:|---:|
| Standard Embeddings | TBD | TBD | TBD | TBD |
| Factorized Embeddings | TBD | TBD | TBD | TBD |

## Planned Figures

- BERT vs ALBERT parameter count
- BERT vs ALBERT task performance
- Parameter sharing: performance by configuration
- Parameter sharing: parameter count by configuration
- Standard vs factorized embeddings: parameter count and performance

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── finetune.py
│   ├── parameter_count.py
│   ├── model_configs.py
│   └── plot_results.py
├── configs/
│   ├── reproduction/
│   ├── parameter_sharing/
│   └── factorized_embeddings/
├── results/
│   ├── reproduction_results.csv
│   ├── parameter_sharing_results.csv
│   └── embedding_results.csv
└── plots/
