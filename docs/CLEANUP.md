# Focused cleanup

The active project now does only four things: load the original credit-card CSV, generate training datasets, train prediction models and compare their scores.

Removed from the active code and reports: the extra dataset loader, unrelated analyses, additional oversampling controls, probability correction, auxiliary diagnostics and bootstrap reporting. Old documents and artifacts are preserved under `archive/out-of-scope-2026-11-09/`, outside the supported pipeline. They are not included in the current report.

The retained comparison contains five requested generation variants and the original-data baseline. The earlier scope cleanup reused valid saved fits. For the current grounded report, the focused comparison is rerun from scratch with candidate-validation predictions, batch-loss traces, exact classifier parameters and clock events saved as evidence. Earlier attempts are archived separately. The focused export drops unrelated saved fields and outputs and supplies complete `training.csv.gz` files alongside generated-only `synthetic.csv.gz` files.

Shared preprocessing, validation-only model/threshold selection, efficient threshold calculation, source snapshots, compressed classifier files and artifact verification remain because they support this comparison directly. Verification checks generated categories/counts, complete training-table contents, duplicate-safe splits, saved predictions and recomputed performance metrics.

Run the tests, verifier and report generator using the commands in the README. No changes have been pushed.

Accepted fresh-run checks: 15 unit/regression tests passed; lint and formatting passed. Verification replayed 120 classifiers, independently checked 240 result rows, and checked 251,670 generated records and all 30 complete training tables. It also recomputed 55 validation-selection APs, reconstructed 3,000 epoch totals from 23,400 batch records, and matched 150 recorded clock intervals. The small preflight run's scores are excluded from the report.

The evidence-logging preflight caught and corrected a variable-name collision that caused a second checkpoint to fail. Preliminary runs affected by that instrumentation error are excluded. The corrected preflight verifies both checkpoints before accepting a full rerun. The report now refuses modified result/evidence inputs, and tests check count formulas, tied-score AP/ROC contributions and this report integrity guard. See the final verification and quality-check JSON files for accepted-run counts.
