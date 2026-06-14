# Experiment 2: Parameter Sharing Ablation

Owner: Person responsible for the parameter-sharing experiment.

This folder contains the script for comparing ALBERT-style parameter sharing configurations.

Implementation note: Hugging Face ALBERT supports full/grouped sharing through `num_hidden_groups`. This project implements attention-only and FFN-only sharing by constructing separate ALBERT layer groups and tying the selected submodules across groups.

Parameter counts only:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py
```

Optional training from random initialization:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train
```

Results are written to:

```text
results/parameter_sharing_results.csv
```
