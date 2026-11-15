# Review of the current project

Reviewed on 13 November 2026. This review covers the active source, saved run metadata, saved metrics and the new Colab preparation. No experiments were launched during this preparation.

## Completed work

The project uses `UCI_Credit_Card.csv`, with 30,000 customers, 23 predictor columns and the next-month default target. Customer ID is excluded from learning. Five saved approximately 60/20/20 splits keep identical feature groups in one partition. All methods within a split use the same original training, validation and test records.

The saved generation matrix contains the real-data baseline and five augmented variants. AGSS interpolation uses DBSCAN guidance. Plain diffusion has MLP and Transformer variants. AGSS-guided diffusion also has MLP and Transformer variants. The requested Transformer-or-Mamba choice was implemented with Transformer.

Each augmented dataset retains its original training records and adds generated defaults until the two training classes are balanced. Across the five splits, real training sets have 17,996–18,004 rows. Enlarged sets have 27,988–28,180 rows. All 30 complete training tables are available under `results/credit_card/seed_*/METHOD/training.csv.gz`. The 25 augmented tables collectively contain 251,670 generated rows.

The completed prediction stage contains logistic regression, histogram gradient boosting, random forest and extra trees. Its 120 model fits produce 240 scoring rows covering cutoff .5 and a validation-selected F1 cutoff. The existing verification record reports successful replay of all 120 classifiers, recomputation of all 240 scoring rows, 55 generator/density selection AP values and batch-to-epoch training-loss checks. The previous quality-check record reports 15 tests passing. Those historical experiment checks were read during this review.

The saved metrics show that 0 of 20 augmentation/classical-classifier combinations improve mean AP over their matching original-data baseline. This result describes the completed classical comparison. Neural classifier performance is still pending.

## Gap addressed by this preparation

The previous prediction-model factory contains four classical classifiers. The generation-stage MLP and Transformer learn to generate default records. A downstream neural classifier must learn the binary default target from the complete two-class training table.

The new `creditrisk/deep_learning.py` prepares that downstream stage with an MLP classifier and a Transformer classifier. Both train on the real baseline and all five augmented datasets, using the five saved splits. This gives 60 neural fits and 120 future scoring rows. Original training data are the baseline for each classifier on the same split.

MLP and Transformer classifier architectures were selected for this preparation. The MLP has hidden layers 128/64 and dropout .2. The classifier Transformer has two layers, four heads and width 32, with a token for each standardized numeric or one-hot encoded component and mean pooling. This differs from the original-feature tokenization used by the generator. Both classifiers have explicitly documented fixed training recipes. The architectures and budgets have not yet been validated through optimization.

The new script reuses the existing real-training-only preprocessing and metric functions. It reads existing training tables, checks the original CSV hash and saved table hashes, verifies split fingerprints and duplicate isolation, and checks that each table begins with the original real training rows. It fits no generator. The Colab bundle supplies the saved inputs and current local source.

## What needs care when interpreting results

- Augmentation increases the number of default examples and changes the training class balance. The comparison measures their combined effect. It does not identify which part causes any observed score change.
- The generation density settings and diffusion checkpoint were selected using a small histogram-gradient-boosting validation proxy. The neural comparison therefore evaluates those existing selected datasets. It does not tune generation separately for each neural classifier.
- AP measures ranking of defaults and is the existing primary comparison metric. F1, precision and recall also depend on the chosen cutoff. Any small F1 gain needs its corresponding AP and confusion counts to explain the change.
- Repeated partitions overlap, and the dataset/test scheme has been inspected in earlier project iterations. Means and split standard deviations are descriptive exploratory results.
- AGSS mixed-feature interpolation and the diffusion sampler/backbones contain documented project adaptations. The code supports this project comparison, while exact replication of published benchmark results remains unestablished.

## Code and portability findings

The active code separates data handling, generation, classifiers, metrics, artifact checks and report construction. The new neural stage imports shared preprocessing, metrics and artifact utilities. Historical notebooks and earlier broader analyses are under `archive/`. The existing source snapshot in the accepted run preserves the fitting implementation used for that run.

The current local rebuild has substantial uncommitted and untracked files. Cloning the repository URL alone would not reliably reproduce this workspace. The upload zip captures the current files and saved training inputs with a file-by-file SHA-256 manifest.

The existing generator creates CPU tensors and uses CPU multiprocessing. Selecting a Colab GPU does not automatically accelerate that generator. The new neural training entry point explicitly moves classifier tensors/models to CUDA and runs jobs sequentially. Its notebook requests CUDA and reports an error if unavailable.

Completed neural jobs save selected weights, epoch histories, validation/test prediction arrays, metrics and completion hashes. Resuming verifies and skips completed jobs. An interrupted classifier restarts its fit. Exact input, code, configuration and recorded library versions are checked for resume. A new runtime can produce different results, so runtime and GPU details are recorded.

The older verifier, evidence exporter and report builder expect the classical model names and artifact layout. They cannot currently be applied directly to the new neural outputs. The neural script prepares its own report and selected-weight prediction replay. An equivalent full independent neural artifact verifier remains a limitation of the prepared workflow.

## Checks performed now

The no-fit validation command passed on all five splits and all six saved training tables per split. Python source and every notebook code cell were syntax-checked. Lint and formatting checks passed on the new Python files. The notebook JSON was checked for empty outputs, empty execution counts and a disabled training switch. The upload archive was built and checked for integrity.

No neural model fitting, generator rerun, CUDA execution, Colab installation or Drive mounting was performed. The first future run must establish whether optimization, GPU execution and output generation work end to end. The delivered notebook keeps `RUN_TRAINING = False`.

## Relevant files

- `creditrisk/deep_learning.py` prepares neural training, saved-weight replay and the future neural summary/report.
- `colab/Credit_Risk_Deep_Learning.ipynb` prepares upload, hash verification, installation, no-fit input checks, Drive output and explicit training launch.
- `colab/build_bundle.py` packages current source and saved training inputs without fitting.
- `colab/README.md` explains execution, architectures, metrics, evidence and resume behavior.
- `results/credit_card/run.json`, `verification.json`, `quality_checks.json` and `metrics.csv` provide the existing run's settings, historical checks and actual classical results.
- `docs/REPORT.md`, `docs/METRICS.md` and `docs/EVIDENCE.md` explain the completed classical results and their evidence.
