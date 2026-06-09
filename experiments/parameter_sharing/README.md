# Experiment 2: Parameter Sharing Ablation

Owner: Person responsible for the parameter-sharing experiment.

This folder contains the script for comparing ALBERT-style parameter sharing configurations.

Important: Hugging Face ALBERT supports full/grouped sharing through `num_hidden_groups`, but not attention-only or FFN-only sharing directly. Those configurations are marked as TODO and skipped when running with `--continue_on_todo`.

Parameter counts only:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --continue_on_todo
```

Optional training from random initialization:

```bash
python experiments/parameter_sharing/run_parameter_sharing.py --train --continue_on_todo
```

Results are written to:

```text
results/parameter_sharing_results.csv
```
