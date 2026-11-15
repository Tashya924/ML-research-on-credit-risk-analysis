"""Density discovery and a transparent mixed-data AGSS interpolation adaptation."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import classifier_preprocessor


def discover(x, seed, categorical=None, quantile=0.9, min_samples=12):
    encoded = classifier_preprocessor(x, categorical).fit_transform(x)
    z = PCA(n_components=0.85, svd_solver="full", random_state=seed).fit_transform(encoded)
    k = min(min_samples, len(x))
    if k < 2:
        raise ValueError("Insufficient minority records for DBSCAN")
    distances = NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z)[0][:, -1]
    epsilon = max(float(np.quantile(distances, quantile)), 1e-6)
    labels = DBSCAN(eps=epsilon, min_samples=k).fit_predict(z)
    keep = labels >= 0
    if keep.any():
        _, labels[keep] = np.unique(labels[keep], return_inverse=True)
        counts = np.bincount(labels[keep])
        weights = np.sqrt(counts) / np.sqrt(counts).sum()
    else:
        counts, weights = np.array([], dtype=int), np.array([])
    info = dict(
        eps=epsilon,
        min_samples=k,
        distance_quantile=quantile,
        pca_dimensions=z.shape[1],
        cluster_counts=counts.tolist(),
        noise_count=int((~keep).sum()),
        noise_fraction=float((~keep).mean()),
        cluster_probabilities=weights.tolist(),
    )
    return labels, weights, info


def interpolate(minority, categorical, labels, count, seed, curvature=0.05):
    """Within-cluster neighbors, perpendicular numeric perturbation, parent categories.

    Curvature magnitude and mixed-feature decoding are stated project choices:
    the AGSS paper does not specify enough details for bit-for-bit reproduction.
    """
    rng = np.random.default_rng(seed)
    clusters = sorted(set(labels) - {-1})
    if not clusters:
        return minority.iloc[:0].copy()
    numeric = [c for c in minority if c not in categorical]
    scaler = StandardScaler().fit(minority[numeric])
    z = scaler.transform(minority[numeric])
    preprocessor = classifier_preprocessor(minority, categorical).fit(minority)
    frames = []
    quotas = rng.multinomial(count, np.ones(len(clusters)) / len(clusters))
    for cluster, quota in zip(clusters, quotas, strict=False):
        if quota == 0:
            continue
        ids = np.flatnonzero(labels == cluster)
        encoded = preprocessor.transform(minority.iloc[ids])
        neighbors = (
            NearestNeighbors(n_neighbors=min(4, len(ids)))
            .fit(encoded)
            .kneighbors(encoded, return_distance=False)
        )
        base = rng.integers(len(ids), size=quota)
        other = np.array(
            [rng.choice(neighbors[i][neighbors[i] != i]) if len(neighbors[i]) > 1 else i for i in base]
        )
        p, q = ids[base], ids[other]
        direction = z[q] - z[p]
        orthogonal = rng.normal(size=direction.shape)
        norm_squared = (direction**2).sum(axis=1, keepdims=True)
        projection = np.divide(
            (orthogonal * direction).sum(axis=1, keepdims=True),
            norm_squared,
            out=np.zeros_like(norm_squared),
            where=norm_squared > 0,
        )
        orthogonal -= projection * direction
        norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
        orthogonal = np.divide(orthogonal, norm, out=np.zeros_like(orthogonal), where=norm > 0)
        alpha = rng.random((quota, 1))
        perturbed = (
            z[p] + alpha * direction + curvature * np.sin(np.pi * alpha) * np.sqrt(norm_squared) * orthogonal
        )
        frame = pd.DataFrame(scaler.inverse_transform(perturbed), columns=numeric)
        for column in numeric:
            frame[column] = frame[column].clip(minority[column].min(), minority[column].max())
            if np.equal(minority[column], np.rint(minority[column])).all():
                frame[column] = np.rint(frame[column])
        for column in categorical:
            frame[column] = np.where(
                rng.random(quota) < alpha[:, 0], minority.iloc[q][column], minority.iloc[p][column]
            )
        frames.append(frame[minority.columns])
    return pd.concat(frames, ignore_index=True)
