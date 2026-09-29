"""
Adaptive Generative Synthetic Sampling (AGSS).

ADASYN-style difficulty weights, computed on the real training partition only, decide *where*
synthetic defaulters are placed; the rows themselves come from a generative model's pool
(TabDDPM by default). Hard defaulters (surrounded by non-defaulters) receive more synthetic
neighbours than easy ones. Non-defaulters are drawn uniformly from the pool.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

TARGET_COL = "default.payment.next.month"


def adasyn_weights(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, k: int = 5) -> np.ndarray:
    """
    Difficulty weight of every real defaulter: share of non-defaulters among its k nearest
    training neighbours (He et al., 2008), normalised to sum to 1.
    """
    X = StandardScaler().fit_transform(X_train_raw)
    y = np.asarray(y_train_raw)
    minority = np.where(y == 1)[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neigh = nn.kneighbors(X[minority], return_distance=False)[:, 1:]  # drop the point itself
    r = (y[neigh] == 0).mean(axis=1)
    return r / r.sum() if r.sum() > 0 else np.full(len(minority), 1 / len(minority))


class AGSSSampler:
    """
    Pre-computes, once, which real defaulter every pooled synthetic defaulter is closest to,
    so that samples of any size / default ratio can be drawn cheaply.
    """

    def __init__(
        self,
        X_train_raw: pd.DataFrame,
        y_train_raw: pd.Series,
        pool_df: pd.DataFrame,
        k: int = 5,
        random_state: int = 42
    ):
        self.features = list(X_train_raw.columns)
        self.random_state = random_state
        self.weights = adasyn_weights(X_train_raw, y_train_raw, k=k)

        scaler = StandardScaler().fit(X_train_raw)
        seeds = scaler.transform(X_train_raw[np.asarray(y_train_raw) == 1])
        self.pool_def = pool_df[pool_df[TARGET_COL] == 1].reset_index(drop=True)
        self.pool_non = pool_df[pool_df[TARGET_COL] == 0].reset_index(drop=True)

        print(f"[*] AGSS: assigning {len(self.pool_def):,} pooled defaulters to {len(seeds):,} real defaulters...")
        dist, owner = NearestNeighbors(n_neighbors=1).fit(seeds).kneighbors(
            scaler.transform(self.pool_def[self.features])
        )
        # For every real defaulter, its pooled rows ordered from closest to farthest
        order = np.lexsort((dist.ravel(), owner.ravel()))
        owners_sorted = owner.ravel()[order]
        starts = np.searchsorted(owners_sorted, np.arange(len(seeds)))
        ends = np.searchsorted(owners_sorted, np.arange(len(seeds)), side="right")
        self.buckets = [order[s:e] for s, e in zip(starts, ends)]

    def sample(self, num_defaults: int, num_non_defaults: int) -> pd.DataFrame:
        if num_defaults > len(self.pool_def) or num_non_defaults > len(self.pool_non):
            raise ValueError(
                f"AGSS pool too small: need {num_defaults}/{num_non_defaults}, "
                f"have {len(self.pool_def)}/{len(self.pool_non)} (defaults/non-defaults)"
            )
        rng = np.random.default_rng(self.random_state)
        quota = rng.multinomial(num_defaults, self.weights)

        picked, spare = [], []
        for bucket, q in zip(self.buckets, quota):
            picked.append(bucket[:q])
            spare.append(bucket[q:])
        picked = np.concatenate(picked)
        shortfall = num_defaults - len(picked)
        if shortfall > 0:
            # Seeds whose bucket was too small: top up with the remaining pooled defaulters
            picked = np.concatenate([picked, rng.choice(np.concatenate(spare), shortfall, replace=False)])

        s1 = self.pool_def.iloc[picked]
        s0 = self.pool_non.sample(n=num_non_defaults, random_state=self.random_state)
        return pd.concat([s1, s0], axis=0).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)


def make_agss_generator(sampler: AGSSSampler):
    """
    Wraps a sampler in the generator signature used by the size / ratio benchmarks.
    """
    def generate(total_samples: int, default_ratio: Optional[float] = None) -> pd.DataFrame:
        ratio = 0.5 if default_ratio is None else default_ratio
        num_defaults = int(total_samples * ratio)
        return sampler.sample(num_defaults, total_samples - num_defaults)
    return generate
