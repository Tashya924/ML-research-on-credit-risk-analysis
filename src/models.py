"""
Model definitions for Classical ML, Ensembles, Deep MLP, and PyTorch TabularTransformer.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of all 16 machine learning classification models.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Ridge Classifier": CalibratedClassifierCV(RidgeClassifier(random_state=random_state)),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Extra Tree": ExtraTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(random_state=random_state, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=random_state),
        "Gaussian Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP Classifier": MLPClassifier(random_state=random_state, max_iter=1000),
        "XGBoost": xgb.XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
    }

    if HAS_LGBM:
        models["LightGBM"] = lgb.LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1)

    return models


def get_paper_replication_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the 10 models evaluated in Table 2 of Xu et al. (2024),
    configured to replicate the published 5-fold CV benchmarks.
    """
    models = {
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state, max_depth=12, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, max_depth=8),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "LDA": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000, C=0.0001),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(50,), random_state=random_state, max_iter=200),
        "Gaussian Naive Bayes": GaussianNB(),
    }
    if HAS_LGBM:
        models["LightGBM"] = lgb.LGBMClassifier(random_state=random_state, max_depth=3, num_leaves=8, learning_rate=0.02, verbose=-1, n_jobs=-1)
    return models


def get_deep_mlp(random_state: int = 42) -> MLPClassifier:
    """
    Returns a deep Multi-Layer Perceptron with 3 hidden layers (128, 64, 32).
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=15
    )


class TabularTransformer(nn.Module):
    """
    PyTorch multi-head self-attention transformer encoder for tabular features.
    Projects inputs into an embedding space, passes through transformer encoder layers,
    and applies an MLP classification head.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_features) -> (batch_size, 1, d_model)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)


class PyTorchModelWrapper:
    """
    Scikit-learn compatible wrapper for PyTorch tabular neural networks.
    Provides fit() and predict_proba() APIs.
    """
    def __init__(
        self,
        model_class: Any,
        input_dim: int,
        epochs: int = 15,
        lr: float = 0.001,
        batch_size: int = 512,
        device: Optional[str] = None
    ):
        self.model_class = model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model_class(self.input_dim).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tensor = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X,
            dtype=torch.float32
        )
        y_tensor = torch.tensor(
            y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y,
            dtype=torch.float32
        ).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            pos_probs = self.model(X_tensor).cpu().numpy().flatten()
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
