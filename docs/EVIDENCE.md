# Evidence: from the original CSV to every reported score

## Exact input and splits

The sole input is `UCI_Credit_Card.csv`; SHA-256 is `a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1`. ID is excluded from model features. Customer IDs appear only in the readable evidence export to locate the original record.

| seed | partition | rows | defaults | nondefaults | default_share |
| --- | --- | --- | --- | --- | --- |
| 101 | train | 17996 | 3974 | 14022 | 0.2208 |
| 101 | validation | 5998 | 1350 | 4648 | 0.2251 |
| 101 | test | 6006 | 1312 | 4694 | 0.2184 |
| 202 | train | 18002 | 3969 | 14033 | 0.2205 |
| 202 | validation | 5999 | 1340 | 4659 | 0.2234 |
| 202 | test | 5999 | 1327 | 4672 | 0.2212 |
| 303 | train | 18001 | 3911 | 14090 | 0.2173 |
| 303 | validation | 6001 | 1374 | 4627 | 0.2290 |
| 303 | test | 5998 | 1351 | 4647 | 0.2252 |
| 404 | train | 18004 | 3974 | 14030 | 0.2207 |
| 404 | validation | 5996 | 1366 | 4630 | 0.2278 |
| 404 | test | 6000 | 1296 | 4704 | 0.2160 |
| 505 | train | 18001 | 4007 | 13994 | 0.2226 |
| 505 | validation | 5998 | 1295 | 4703 | 0.2159 |
| 505 | test | 6001 | 1334 | 4667 | 0.2223 |

Each seed's `split_indices.npz` gives original zero-based CSV row positions for train, validation and test. The readable export adds `csv_line_number = csv_row_index + 2` because the original CSV has one header line. Duplicate feature groups cannot cross partitions. Generated rows never enter validation or test. The verifier checks these mappings against the original CSV and separately refits the training-only preprocessor to check its parameters.

## Worked example selected in advance

This example uses seed **101**, **original training data**, **histogram gradient boosting**, and **validation-selected cutoff 0.33**. It is a concrete row of evidence, not a best-performing example selected from the test results.

| Metric | Value | Actual calculation |
| --- | --- | --- |
| average_precision | 0.5362 | sum(ranking_curve.ap_contribution) |
| roc_auc | 0.7721 | sum(ranking_curve.roc_area_contribution) |
| f1 | 0.5103 | 2*647/(2*647+577+665) |
| precision | 0.5286 | 647/(647+577) |
| recall | 0.4931 | 647/(647+665) |
| accuracy | 0.7932 | (647+4117)/6006 |
| balanced_accuracy | 0.6851 | 0.5*(647/(647+665)+4117/(4117+577)) |
| tn | 4117.0000 | 4117 |
| fp | 577.0000 | 577 |
| fn | 665.0000 | 665 |
| tp | 647.0000 | 647 |

The count-based expressions above show exactly why each rate takes its value. AP and ROC-AUC instead come from the complete ordering of the scores; one cutoff's TP/FP/FN/TN cannot explain them alone.

The key `hist_gradient_boosting` means scikit-learn's `HistGradientBoostingClassifier`, not `GradientBoostingClassifier`. Its final model uses 150 iterations, learning rate .07 and at most 20 leaves. The selection proxy uses 80 iterations, .08 and 15 leaves; its validation AP must not be confused with the final classifier's test AP. Logistic regression allows up to 1,000 iterations; random forest and extra trees each use 150 trees with minimum leaf size 3. All defaults and seeds, not just these headline settings, are in the saved parameter JSON and frozen classifier code.

Follow this example's files:

1. [Original-row split mapping](../results/credit_card/seed_101/split_indices.npz), key `test`, identifies the real records.
2. [Complete training table](../results/credit_card/seed_101/real/training.csv.gz), [fitted preprocessor](../results/credit_card/seed_101/preprocessor.joblib), [classifier parameters](../results/credit_card/seed_101/real/classifier_parameters.json) and [fitted classifier](../results/credit_card/seed_101/real/hist_gradient_boosting.joblib) establish what was trained.
3. [Full-precision prediction arrays](../results/credit_card/seed_101/real/hist_gradient_boosting_predictions.npz), keys `test_y` and `test_probability`, are the score source.
4. [Readable per-record decisions](../results/credit_card/evidence/seed_101/real/hist_gradient_boosting/test_predictions.csv.gz) show source CSV row/ID, truth, score, and decisions at both cutoffs.
5. [All validation cutoff candidates](../results/credit_card/evidence/seed_101/real/hist_gradient_boosting/validation_thresholds.csv) explain why this cutoff was selected.
6. [Ranking contributions](../results/credit_card/evidence/seed_101/real/hist_gradient_boosting/ranking_curve.csv.gz) explain the AP and ROC-AUC sums.
7. [Timing events](../results/credit_card/seed_101/real/events.jsonl) explain the recorded build/fit durations.

The same structure exists for every other method, classifier and split. [The complete metric ledger](../results/credit_card/metric_evidence.csv) contains 2640 individual metric values and their paths/hashes. [The summary ledger](../results/credit_card/summary_evidence.csv) maps all 168 aggregate score entries to their 5 source evidence IDs. [The validation-selection ledger](../results/credit_card/selection_evidence.csv) provides the candidate-setting AP values and their raw validation predictions.

To independently reproduce the example's F1 from the repository root:

```python
import numpy as np
p = np.load('results/credit_card/seed_101/real/hist_gradient_boosting_predictions.npz')
y, score, cutoff = p['test_y'], p['test_probability'], float(p['threshold'])
predicted = score >= cutoff
tp = ((y == 1) & predicted).sum()
fp = ((y == 0) & predicted).sum()
fn = ((y == 1) & ~predicted).sum()
print(2 * tp / (2 * tp + fp + fn))
```

To reconstruct the ranking measures from their saved contributions:

```python
import pandas as pd
curve = pd.read_csv('results/credit_card/evidence/seed_101/real/hist_gradient_boosting/ranking_curve.csv.gz')
print(curve['ap_contribution'].sum())
print(curve['roc_area_contribution'].sum())
```

## What was rerun and checked

This accepted run regenerated all five synthetic-data variants on the original dataset's five splits, retrained all four classifiers on each complete training dataset and the real-only baseline, and regenerated the validation selection scores. Nothing in the current report is copied from the broader archived comparison. A preliminary logging-only attempt was excluded after a variable-name collision; its files are archived and no scores from it enter this report.

[verification.json](../results/credit_card/verification.json) records model replay, independent scikit-learn checks of our count formulas, raw-label/split checks, complete training-table checks, generator batch-to-epoch loss reconstruction, validation-selection checks and timing checks. [run.json](../results/credit_card/run.json) records command/configuration/versions and hashes of the exact fitting source under `source/`. Later reporting code is hashed in [evidence.json](../results/credit_card/evidence.json).

Labels, prediction arrays and source-row mappings are checked, not inferred from old rounded tables. The report builder refuses changed metric/evidence inputs. File integrity and reproducibility do not by themselves prove a generator is better; that conclusion must follow the matched test results, within the stated settings and exploratory limits.
