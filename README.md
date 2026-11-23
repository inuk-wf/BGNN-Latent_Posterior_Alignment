# BGNN: Latent-Posterior Alignment

Code accompanying **Hidden Axis of Uncertainty: Latent-Posterior Alignment in Graph Neural Networks with Bayesian Output Layers** ([paper](https://arxiv.org/abs/2608.20758)).

Molecular GIN/BGNN training, predictive uncertainty estimation, Latent-Posterior Alignment Score (LPAS), and Density Uncertainty Criterion (DUC).

## Installation

Use Python 3.12. Run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Dependencies are specified in `pyproject.toml`. CPU is the default; CUDA requires a compatible PyTorch installation and GPU driver.

## Data

Place prepared data under `data/local/` or pass another path with `--data`. Research datasets and pretrained models are not bundled.

CSV input requires `smiles,target` columns, one molecule and one finite scalar target per row. At least six molecules are required; smaller training fractions may require more. Supply targets in the intended units and with the required transformations already applied.

| Configuration | Prepared target |
|---|---|
| `baseline_bp.json` | Boiling point |
| `baseline_mp.json` | Melting point |
| `baseline_logP.json` | Log partition coefficient |
| `baseline_logPvap.json` | Log vapor pressure, using the intended log base and pressure unit |
| `baseline_H_bonds.json` | Enthalpy-based bond-energy target |
| `baseline_G_bonds.json` | Gibbs-energy-based bond-energy target |

The loader does not apply logarithms, convert units, deduplicate records, or assemble molecular/atomic energy targets. Predictions are returned in input-target units.

Tensor graph files written by `bgnn.data.save_dataset(graphs, path)` are also supported. Each graph stores `smiles`, scalar `y`, 37-column `x`, `edge_index` of shape `[2, n_edges]`, 7-column `edge_attr`, and `charge`. Feature ordering is defined in `bgnn/features.py`.

## Training

```bash
python -m bgnn.training --config configs/baseline_logP.json --data data/local/logP.csv --output outputs/logP_baseline --device cpu
```

Use `--device cuda` for GPU training. Use `--dry-run` with a configuration to display settings without training. Choose a new output directory for each experiment.

Each baseline uses 29 runs (seeds 1-29), a five-layer GIN, 128 hidden channels, 64 latent dimensions, dropout 0.3, and 100 Monte Carlo draws. Each GNN/BGNN stage runs for up to 500 epochs with loss patience 60.

The split reserves 20% for testing and 10% of the remainder for validation. `split.train_fraction` selects a fraction of the remaining training pool. Each run trains its own deterministic GNN and uses its weights to initialize the BGNN. The output-weight and bias priors have zero mean and standard deviation `fit.scale_prior`; the baseline value is 1. Warm start initializes the model parameters independently of the prior.

| Setting | Meaning |
|---|---|
| `n_runs`, `seed_start` | Number of runs and first seed |
| `fit.n_epoch`, `fit.lr` | Maximum epochs and learning rate |
| `fit.n_rand` | Monte Carlo draws |
| `fit.n_batch` | Batch size; `null` selects 64, 256, or 1024 from the input count |
| `fit.scale_prior` | Prior standard deviation |
| `fit.scale_kl`, `fit.scale_loss` | KL weight and likelihood/KL loss multiplier |
| `fit.z_regularize_opt` | `null`, `L1`, `L2`, `anti_alignment`, or `agl` |

## Analysis and plotting

```bash
python -m bgnn.analysis outputs/logP_baseline --output outputs/logP_summary.json
python -m bgnn.plotting outputs/logP_summary.json --output outputs/logP_summary.png
```

Analysis reads test predictions from all configured runs and reports MAE, mean predictive standard deviation, mean posterior standard deviation, LPAS, and ECE. Add `--duc` to include KDE-based DUC. The plot displays metric means with across-run standard deviations.

## Output files

Each experiment contains `experiment.json` and `run_001/` through `run_029/`. Every run saves:

- `split.json`: input row indices for train, validation, and test.
- `checkpoint_GNN.pt` and `gnn_log.npz`: deterministic warm-start model and training log.
- `checkpoint_BGNN.pt` and `log.npz`: Bayesian model and training log.
- `predictions_train.npz`, `predictions_validate.npz`, `predictions_test.npz`: predictions and latent vectors.
- `completed.json`: run ID, seed, and settings.

Prediction fields are `row_ids`, `y`, `y_mean`, `sigma_y`, `z`, `mu_w`, `sigma_w`, and format metadata. Load an inference checkpoint with `bgnn.artifacts.load_model(path)`. GNN outputs are saved when warm start is enabled.

## Code structure

| File | Role |
|---|---|
| `bgnn/models.py` | GIN/BGNN, objectives, and training loop |
| `bgnn/training.py` | Run orchestration and prediction export |
| `bgnn/features.py`, `bgnn/data.py` | Molecular features, input, and splitting |
| `bgnn/artifacts.py`, `bgnn/runtime.py` | Checkpoints, seeds, and output storage |
| `bgnn/metrics.py`, `bgnn/analysis.py` | Metrics and across-run aggregation |
| `bgnn/plotting.py` | Summary graphs |
| `configs/` | Six target-specific baseline configurations |

## Citation

Please cite the [paper](https://arxiv.org/abs/2608.20758) when using this code. Citation metadata are provided in [CITATION.cff](CITATION.cff).
