"""Evaluation shared by training, reporting, and independent artifact checks."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLDS = np.linspace(0.05, 0.95, 91)


def threshold(y, probabilities):
    """Vectorized fixed-grid F1 selection; ties choose the lowest threshold."""
    y = np.asarray(y, dtype=bool)
    predictions = np.asarray(probabilities)[:, None] >= THRESHOLDS
    tp = (predictions & y[:, None]).sum(axis=0)
    denominator = predictions.sum(axis=0) + y.sum()
    scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    return float(THRESHOLDS[np.argmax(scores)])


def metrics(y, probabilities, cutoff):
    y = np.asarray(y, dtype=bool)
    predictions = np.asarray(probabilities) >= cutoff
    tp = int((predictions & y).sum())
    fp = int((predictions & ~y).sum())
    fn = int((~predictions & y).sum())
    tn = int((~predictions & ~y).sum())

    def divide(a, b):
        return a / b if b else 0.0

    return dict(
        average_precision=average_precision_score(y, probabilities),
        roc_auc=roc_auc_score(y, probabilities),
        f1=divide(2 * tp, 2 * tp + fp + fn),
        precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn),
        accuracy=divide(tp + tn, len(y)),
        balanced_accuracy=(divide(tp, tp + fn) + divide(tn, tn + fp)) / 2,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )
