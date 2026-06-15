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

## MRPC Fine-Tuning

Run all MRPC parameter-sharing configurations:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train \
  --config configs/parameter_sharing/no_sharing_mrpc.yaml \
  --config configs/parameter_sharing/shared_attention_mrpc.yaml \
  --config configs/parameter_sharing/shared_ffn_mrpc.yaml \
  --config configs/parameter_sharing/full_sharing_mrpc.yaml
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
