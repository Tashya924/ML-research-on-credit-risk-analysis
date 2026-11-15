# What every metric means, why it is used, and where it comes from

## Common basis

Every prediction-performance score uses **real test records**, with `default.payment.next.month=1` as the positive class. A predicted default means `score >= threshold`. Scores are binary class-1 measures, not weighted or macro F1. For precision, recall and F1, a zero denominator is reported as zero. That convention is explicit in the frozen metric code.

The two policies are `raw_0.5` (fixed cutoff .5) and `validation_f1` (cutoff selected on real validation records). AP and ROC-AUC use the same unthresholded score vector under both policies, so their repetition is intentional. Two policy rows do not represent two model fits.

| CSV field | Meaning | Formula | Why included | Uses cutoff? |
| --- | --- | --- | --- | --- |
| average_precision | AP | Sum of precision × increase in recall at each distinct descending score. | Primary ranking measure: gives attention to the default class when it is the minority. | No |
| roc_auc | ROC-AUC | Sum of trapezoid areas under TPR versus FPR; tied positive/negative scores count half. | Complementary ranking measure; does not replace precision under class imbalance. | No |
| f1 | Default-class F1 | 2 TP / (2 TP + FP + FN) | Balances default precision and recall at a stated decision cutoff. | Yes |
| precision | Precision | TP / (TP + FP) | Of the flagged defaults, how many actually defaulted? | Yes |
| recall | Recall | TP / (TP + FN) | Of the actual defaults, how many were caught? | Yes |
| accuracy | Accuracy | (TP + TN) / N | Overall correctness; can conceal weak default detection when non-defaults dominate. | Yes |
| balanced_accuracy | Balanced accuracy | 0.5 × [TP/(TP+FN) + TN/(TN+FP)] | Weights the two classes equally rather than by their frequency. | Yes |
| tn | True negatives | Count(y=0 and score < cutoff) | Non-defaults correctly left unflagged. | Yes |
| fp | False positives | Count(y=0 and score >= cutoff) | Non-defaults incorrectly flagged. | Yes |
| fn | False negatives | Count(y=1 and score < cutoff) | Defaults missed. | Yes |
| tp | True positives | Count(y=1 and score >= cutoff) | Defaults correctly flagged. | Yes |

The implementation is in [the exact fitting-time metric code](../results/credit_card/source/creditrisk/metrics.py). Definitions agree with scikit-learn's [AP](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html), [ROC-AUC](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.roc_auc_score.html), [precision/recall/F1](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html), and [balanced accuracy](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.balanced_accuracy_score.html) documentation. All seven score values are on a 0–1 scale, confusion counts are integers, and durations are seconds. Those links explain definitions; the local run's library versions and predictions establish its numerical results. AP is the precision-weighted sum of recall increments, **not trapezoidal PR-AUC**. ROC-AUC uses ROC trapezoids and cannot be reconstructed from one confusion matrix.

## Where to check each number

[metric_evidence.csv](../results/credit_card/metric_evidence.csv) has one row per individual metric value. Search by `seed/method/model/policy/metric`. It gives the substituted-count formula or ranking-curve sum, the exact source prediction/model/training files and hashes, the real test size/default count, the code file, and the physical line in `metrics.csv`. The authoritative full-precision values are in NPZ arrays; the CSVs are readable derivatives.

`ranking_curve.csv.gz` groups tied scores at their common threshold. Sum `ap_contribution` to recover AP; sum `roc_area_contribution` to recover ROC-AUC. The curve also shows cumulative TP/FP, precision, recall and FPR. Both sums are checked against scikit-learn. [A worked example is here](EVIDENCE.md).

## Cutoff selection and generator-setting scores

For `validation_f1`, 91 cutoffs from .05 through .95 in .01 steps are evaluated on validation predictions. The first maximum is chosen, so ties select the lowest cutoff. Each model's `validation_thresholds.csv` contains all 91 confusion counts and F1 values. This selects an operating cutoff, not a better ranking model.

[selection_evidence.csv](../results/credit_card/selection_evidence.csv) covers 55 saved validation AP scores used for DBSCAN-rule and generator-checkpoint selection. They are **not test results**. A fixed small gradient-boosting proxy is fitted on real training plus candidate-generated records and scored on real validation records. Its exact parameters are in each seed's `selection_model.json`; the final four classifiers have separate parameters in `classifier_parameters.json`. Maximum validation AP selects the density rule and checkpoint; test scores do not select either. Checkpoint candidates and failures are never substituted with test scores.

## Means, standard deviations and differences

For each method/classifier, the report uses the unweighted arithmetic mean of the 5 split scores, taking only one policy at a time. Sample SD is `sqrt(sum((score-mean)^2)/(n-1))`, with `n=5`; it is not a confidence interval or standard error. We do not pool predictions across overlapping splits or count the two policy rows as additional runs.

[summary_evidence.csv](../results/credit_card/summary_evidence.csv) lists every reported mean, SD, sample count, and the 5 individual evidence IDs from which it was computed. `comparison_summary.csv` is the wide version. `paired_differences.csv` subtracts the same classifier's real-only score from each generated-data score **within the same seed/test partition**. Positive AP/F1 differences favor augmentation. These are descriptive differences, not significance tests.

## Counts, time and configuration fields

[metadata_evidence.csv](../results/credit_card/metadata_evidence.csv) explains and locates each non-score numeric field in `metrics.csv`:

- `threshold`: the stated cutoff; its justification is fixed .5 or maximum validation F1.
- `real_train_rows`, `synthetic_rows`, `train_rows`, `test_rows`: counts of the corresponding data tables/indices. Complete training rows equal real training plus generated defaults.
- `generation_seconds`: the observed method-build wall-time interval. It includes generator fitting, sampling, validation proxy selection, logging and generator-file exports. It is **not pure sampling speed**.
- `fit_seconds`: the observed classifier-fit interval including its start-event logging. It excludes preprocessing, score prediction and cutoff search.

Times are clock differences from `events.jsonl`, checked against the CSV. Parallel workers share the machine, so timings are operational observations, not isolated hardware benchmarks. The same fit/build time appears in both policy rows and must not be summed twice. `seed`, `method`, `model` and `policy` are identifiers, not performance metrics. `run.json` records the actual command, versions, worker/thread counts and start/end times.

## Generator training loss is not test performance

`training_loss.csv` records each epoch's optimization objective. It is the example-weighted mean of batch `numeric_mse + categorical_vb`; `batch_loss.csv` records the actual batch components and example counts so the epoch totals can be recomputed. Numeric loss is Gaussian noise-prediction MSE; categorical loss is the official TabDDPM stochastic multinomial variational loss, normalized by the number of categorical features. It is not the earlier CE-only shortcut.

The exact computation is in [the frozen diffusion adapter](../results/credit_card/source/creditrisk/diffusion.py) and [the frozen upstream diffusion core](../results/credit_card/source/vendor/tab_ddpm/gaussian_multinomial_diffsuion.py). Random corruption is part of training; batch logs establish the recorded optimization trace, not a replay from final weights alone. Lower training loss is not proof of useful synthetic records and does not select the checkpoint: validation AP does. A selected epoch-50 model can coexist with a loss history through epoch 150 because both candidate budgets were trained and compared.

DBSCAN epsilon, distance quantile, cluster counts/noise, PCA dimensions, generator epochs/timesteps and model hyperparameters describe the generation recipe. They are not prediction-performance scores. Their exact values and selection evidence are saved with the corresponding seed/method. Fixed constants such as PCA .85 and curvature .05 are project choices, not empirically proven optima.
