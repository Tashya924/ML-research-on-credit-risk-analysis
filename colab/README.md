# Colab preparation

Training has not been run for the new deep learning classifiers. The notebook is delivered with no cell outputs or execution counts, and `RUN_TRAINING = False`.

## Open and use

1. Open `Credit_Risk_Deep_Learning.ipynb` in Google Colab and select a GPU runtime.
2. Run the upload cell and upload `credit_risk_colab.zip` from this folder. It contains the current local source, original CSV and all 30 saved training tables with their split mappings. Its manifest checks every bundled file before installation.
3. Run the dependency and input-validation cells. Validation reads the tables and checks their hashes, split identities, real-training prefixes, augmented row counts and duplicate-group isolation. It fits no models.
4. Mount Drive so classifier outputs survive a runtime disconnect. Set `RUN_TRAINING = True` only when ready to start. The training cell launches 60 fits, two classifiers across six datasets and five saved splits.
5. After interruption, reconnect, upload the same bundle, mount the same Drive, and set `RESUME = True`. Completed classifier jobs are hash-checked and skipped. An interrupted classifier restarts its fit from epoch one. Code, input, configuration and recorded library versions must match for resume.
6. The final cell displays the generated neural comparison report after training finishes. Outputs are under the configured Drive `OUTPUT` directory.

The package uses saved generated datasets from the accepted run. The existing generation command, when explicitly desired, remains `python train.py --output results/new_generation`. It also trains the existing four classical classifiers and currently uses CPU tensors. Its fresh output can be supplied to the neural script with `--input`. The Colab notebook defaults to the existing saved datasets.

## Neural classifier recipes

- `mlp_classifier` uses hidden layers 128 and 64, ReLU and dropout 0.2.
- `transformer_classifier` uses a separate learned token for each standardized numerical or one-hot encoded component, width 32, four heads, two encoder layers, dropout 0.2 and mean pooling. This is the classifier's explicit project architecture. The generation Transformer uses a different representation and objective.
- Both use AdamW with learning rate 0.001, weight decay 0.0001, batch size 512, gradient norm clipping at 1, and unweighted binary cross-entropy. Maximum 100 epochs, early stopping after 15 epochs without improved validation AP. Exact ties retain the earlier epoch.
- Scaling and categorical encoding fit only the original real training partition, then transform all six training variants and the real validation/test partitions. Original customer ID is excluded. Real training records form the prefix of each augmented table.
- Each classifier is initialized with the same split seed across methods. Class weighting and replacement sampling are absent from the training recipe. The augmented datasets already balance the classes. This comparison measures the combined effect of adding the generated defaults and balancing the training distribution.
- Validation AP chooses the model epoch. Validation F1 chooses a cutoff from .05 through .95 in .01 steps. Test scores are recorded at that cutoff and .5. The same existing metric implementation is reused.
- Sequential fits use CUDA when available. The notebook requests CUDA explicitly so a missing GPU produces a clear error. The CLI also supports CPU. Deterministic algorithms and a CUDA workspace configuration are requested. Runtime/library/GPU information is recorded with the results.

## Outputs and evidence after a future run

`run.json` records settings, versions, input hashes and source hashes. `source/` freezes the neural training implementation and reused helpers. Each `seed_*/METHOD/CLASSIFIER/` contains selected `classifier.pt` weights, `history.csv`, `selected_validation.npz`, full `predictions.npz`, per-policy `metrics.csv` and an atomic completion manifest. The prediction arrays include original CSV test row positions. Saved weights are reloaded and their test predictions compared before a job is accepted.

The root `metrics.csv`, `comparison_summary.csv`, `paired_differences.csv` and `REPORT.md` are created when all configured jobs complete. The summary uses means, sample standard deviations and counts over the saved splits. Repeated splits overlap. AP and ROC-AUC use score rankings, while F1, precision, recall, accuracy, balanced accuracy and confusion counts depend on the decision cutoff. The repository's `docs/METRICS.md` explains their formulas and purposes.

The existing `creditrisk.verify`, `creditrisk.evidence` and `creditrisk.report` commands handle the earlier classical run's artifact layout. The neural script creates its own report and checks selected-weight prediction replay. The complete independent classical verification workflow has not yet been extended to the neural output layout.

## Local packaging and checks

From the repository root, `python -m colab.build_bundle` rebuilds the upload zip without training. `python -m creditrisk.deep_learning --validate-only` validates saved inputs without training or writing experiment results. The zip captures local changes, so uploading it avoids depending on whether the rebuilt code has been pushed to GitHub.

The new code has been inspected, parsed, linted and checked against the saved inputs. Neither neural optimization nor CUDA execution has been tested yet, in accordance with the instruction to prepare scripts without running experiments. Colab installation, Drive mounting and runtime behavior also remain untested.
