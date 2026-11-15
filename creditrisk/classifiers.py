"""One source of classifier definitions; no optional silent model substitution."""

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_NAMES = ("logistic_regression", "hist_gradient_boosting", "random_forest", "extra_trees")


def make_models(seed, threads=4):
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.07, max_leaf_nodes=20, early_stopping=False, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3, n_jobs=threads, random_state=seed
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=150, min_samples_leaf=3, n_jobs=threads, random_state=seed
        ),
    }


def selection_model(seed):
    """Fixed small proxy; never evaluates test data during method selection."""
    return HistGradientBoostingClassifier(
        max_iter=80, max_leaf_nodes=15, learning_rate=0.08, early_stopping=False, random_state=seed
    )
