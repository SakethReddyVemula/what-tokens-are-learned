# scripts/slurm/

SLURM batch wrappers for running the full pipeline on a cluster.

| File                         | Purpose                                        |
| ---------------------------- | ---------------------------------------------- |
| `run_pipeline_slurm.sh`      | Submits `scripts/run_pipeline.py` as a batch job |
| `run_pipeline_myte_slurm.sh` | Same, configured for the MyTE tokenizer         |

```bash
sbatch scripts/slurm/run_pipeline_slurm.sh
```

## Before submitting

The `#SBATCH` headers record the exact resources used for the paper's runs.
Adapt them to your cluster — in particular `--nodelist`, `--mail-user`, and the
`source .../bin/activate` line naming a virtualenv path.

Edit the configuration block below the headers to set the grid (`LANGS`,
`VOCAB_SIZES`, `MODEL_TYPES`, `DATASET_SPLIT`).

Both scripts `cd` to the repository root before running, so relative output
directories resolve correctly regardless of the submission directory.
`HF_TOKEN` and `WANDB_API_KEY` must be present in the job environment.
