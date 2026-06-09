# Experiment Notes

## Important limitations

- The reproduction experiment uses pretrained checkpoints.
- The ablation experiments use controlled model configurations.
- Exact pretrained checkpoints for all parameter-sharing and embedding ablation variants may not exist.
- Partial sharing variants, such as shared-attention and shared-FFN only, require custom modeling code and are marked as TODO.
- If ablation training is run, those models are randomly initialized unless explicitly changed.

## Recommended presentation wording

For the ablations, describe the implementation as a representative architectural analysis under limited compute, not as a full reproduction of the paper's large-scale pretraining ablations.
