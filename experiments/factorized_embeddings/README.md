# Experiment 3: Factorized Embeddings

Owner: Person responsible for the factorized-embedding experiment.

This folder contains the script for comparing standard embeddings with ALBERT-style factorized embeddings.

Parameter counts only:

```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py
```

Optional training from random initialization:

```bash
python experiments/factorized_embeddings/run_factorized_embeddings.py --train
```

Results are written to:

```text
results/embedding_results.csv
```
