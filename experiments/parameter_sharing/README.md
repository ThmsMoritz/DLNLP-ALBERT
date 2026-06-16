# Experiment 2: Parameter Sharing Ablation

Owner: Person responsible for the parameter-sharing experiment.

This folder contains the scripts and configuration files used to compare different ALBERT parameter-sharing strategies.

Implementation note: Hugging Face ALBERT supports full/grouped sharing through `num_hidden_groups`. This project additionally implements attention-only and FFN-only sharing by constructing separate ALBERT layer groups and tying the selected submodules across groups.

## Configurations

The following parameter-sharing strategies are evaluated:

* No Sharing
* Shared Attention
* Shared FFN
* Full Sharing (standard ALBERT)

Experiments were conducted on two GLUE tasks:

* SST-2 (sentiment classification)
* MRPC (Microsoft Research Paraphrase Corpus)

## Extension: Layer-Position Parameter Sharing

To further investigate ALBERT's parameter-sharing mechanism, an additional architectural extension was implemented.

Instead of sharing parameters across all transformer layers, parameter sharing is restricted to either the lower or upper half of the network:

* Lower-Half Sharing

  * Layers 1–6 share attention and FFN parameters
  * Layers 7–12 remain independent

* Upper-Half Sharing

  * Layers 1–6 remain independent
  * Layers 7–12 share attention and FFN parameters

This extension evaluates whether the location of parameter sharing influences downstream task performance.

The extension was evaluated on both:

* SST-2
* MRPC

## Parameter Counts Only

Run the parameter-sharing comparison without training:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py
```

## SST-2 Fine-Tuning

Run all SST-2 parameter-sharing configurations:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train
```

To run only the layer-position sharing extension:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train \
  --config configs/parameter_sharing/lower_half_sharing.yaml \
  --config configs/parameter_sharing/upper_half_sharing.yaml
```

## MRPC Fine-Tuning

Run the original MRPC parameter-sharing configurations:

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

## Results

All experiment results are appended to:

```text
results/parameter_sharing_results.csv
```

The results file contains:

* Task name
* Sharing strategy
* Parameter count
* Embedding parameter count
* Accuracy
* F1 score (MRPC)
* Training time
* GPU memory usage

## Summary

This experiment investigates the trade-off between parameter efficiency and downstream task performance in ALBERT. In addition to the standard parameter-sharing ablations, the layer-position sharing extension evaluates whether sharing parameters in lower versus upper transformer layers affects model performance.
