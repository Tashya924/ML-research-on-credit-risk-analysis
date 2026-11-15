# Credit-card synthetic data comparison

Use the existing `UCI_Credit_Card.csv`, generate training datasets with different methods, train prediction models, and compare their performance.

Read the [grounded report](docs/REPORT.md), [metric definitions](docs/METRICS.md), [worked evidence example](docs/EVIDENCE.md), and [full model comparison](docs/COMPARISON.md).

## Methods

- Original training data: baseline.
- AGSS interpolation guided by DBSCAN.
- Diffusion with an MLP denoiser.
- Diffusion with a feature-token Transformer denoiser.
- AGSS + DBSCAN-guided diffusion with MLP.
- AGSS + DBSCAN-guided diffusion with Transformer.

Prediction models: logistic regression, gradient boosting, random forest and extra trees. The Transformer is used in dataset generation. We chose Transformer from the requested Transformer-or-Mamba option.

Each generated training dataset retains the original training rows and adds generated defaults to balance the classes. `synthetic.csv.gz` contains the new rows; `training.csv.gz` contains the complete two-class training dataset. Validation and test records remain real.

## Run

Python 3.10+; run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Short software check using a subset of the original CSV
python train.py --smoke --output results/my_check
python -m creditrisk.verify --input results/my_check

# Full comparison; use a fresh output directory
python train.py --output results/my_comparison
python -m creditrisk.verify --input results/my_comparison
python -m creditrisk.evidence --input results/my_comparison
python -m creditrisk.report --input results/my_comparison --output docs/REPORT.md
```

Defaults: five splits (101/202/303/404/505), diffusion checkpoints at 50/150 epochs, 100 timesteps, batch size 512 and four CPU threads per worker (two independent split workers by default). Checkpoints and F1 cutoffs are chosen on validation data. Test data never selects them. Completed jobs can be resumed with the same code/configuration and `--resume`.

## Delivered results

`results/credit_card/` contains only the requested comparison on the original dataset. The accepted grounded run regenerates the requested datasets and retrains the classifiers from scratch. It saves validation predictions for every candidate setting as well as final test predictions. `run.json` records the command, actual settings/versions, start/end times and frozen fitting source. Earlier reused-fit exports are archived and excluded.

- `metrics.csv`: every classifier, method, split and threshold.
- `comparison_summary.csv`: mean and standard deviation across splits.
- `paired_differences.csv`: changes from each classifier's original-data baseline.
- `dataset_inventory.csv`: paths to generated rows and complete training datasets.
- `seed_*/METHOD/`: datasets, fitted models, generator files and saved predictions.
- `verification.json`: model replay, independent metric, selection-score, loss-total and timing checks.
- `metric_evidence.csv`: every individual score's formula, substituted counts, prediction/model/data paths and hashes.
- `summary_evidence.csv`: the source evidence IDs behind each reported mean and sample SD.
- `selection_evidence.csv`: saved validation APs used for density/checkpoint selection.
- `metadata_evidence.csv`: the source and meaning of count, time and threshold fields.
- `evidence/`: readable real-row predictions, ranking contributions and complete validation cutoff grids.

The [method notes](docs/METHODS.md) describe the generation and evaluation settings. [Cleanup notes](docs/CLEANUP.md) explain the focused code changes.

## Colab deep learning preparation

The [Colab notebook](colab/Credit_Risk_Deep_Learning.ipynb) and [instructions](colab/README.md) prepare MLP and Transformer default classifiers on the saved baseline and augmented datasets. Training is disabled in the delivered notebook. The [current code review](colab/REVIEW.md) distinguishes completed classical results from the prepared neural experiment. Build the upload package from the current local files using `python -m colab.build_bundle`.

The Colab upload package is included at `colab/credit_risk_colab.zip`. It carries the current code and all 30 saved training inputs. The GitHub export also includes the accepted classical run's datasets, prediction arrays, evidence tables and generation checkpoints. Large fitted classical classifier files and historical local archives remain in the original workspace. Full classical model replay requires those fitted classifier files. Neural optimization and Colab GPU execution are pending.
