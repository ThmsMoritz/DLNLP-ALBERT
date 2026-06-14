# Experiment Notes

## Important limitations

- The reproduction experiment uses pretrained checkpoints and fine-tunes them on selected GLUE tasks.
- MNLI uses `validation_matched` by default because the GLUE dataset exposes separate matched and mismatched validation splits.
- The ablation experiments use controlled model configurations.
- Exact pretrained checkpoints for all parameter-sharing and embedding ablation variants may not exist.
- Partial sharing variants, such as shared-attention and shared-FFN only, are implemented by tying selected ALBERT submodules across otherwise separate layer groups.
- If ablation training is run, those models are randomly initialized unless explicitly changed.

## Recommended presentation wording

For the ablations, describe the implementation as a representative architectural analysis under limited compute, not as a full reproduction of the paper's large-scale pretraining ablations.
