"""Protocol and regression tests; no assertions of particular benchmark scores."""

import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from creditrisk.agss import discover, interpolate
from creditrisk.data import TARGET, Schema, load_dataset, partitions
from creditrisk.diffusion import MixedDiffusion
from creditrisk.metrics import THRESHOLDS, metrics, threshold
from vendor.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.dataset = load_dataset()

    def test_duplicate_groups_cannot_cross_splits(self):
        frame = pd.DataFrame(
            {"feature": np.repeat(np.arange(200), 2), TARGET: np.repeat(np.arange(200) % 2, 2)}
        )
        parts, _ = partitions(frame, 42)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            self.assertFalse(set(parts[i].feature) & set(parts[j].feature))

    def test_dataset_target_mapping(self):
        self.assertEqual(self.dataset.frame[TARGET].value_counts().to_dict(), {0: 23364, 1: 6636})
        self.assertEqual(len(load_dataset().categorical), 9)

    def test_schema_rejects_unknown_category(self):
        x = self.dataset.features.iloc[:100]
        schema = Schema().fit(x, categorical=self.dataset.categorical)
        changed = x.copy()
        changed.loc[changed.index[0], "SEX"] = 99
        with self.assertRaises(ValueError):
            schema.encode(changed)

    def test_vector_threshold_matches_sklearn_grid(self):
        rng = np.random.default_rng(3)
        y, p = rng.integers(2, size=500), rng.random(500)
        expected = THRESHOLDS[np.argmax([f1_score(y, p >= t) for t in THRESHOLDS])]
        self.assertEqual(threshold(y, p), expected)
        self.assertAlmostEqual(metrics(y, p, expected)["f1"], f1_score(y, p >= expected))

    def test_interpolation_preserves_categories_and_original_data(self):
        x = self.dataset.features.iloc[:100].copy()
        saved = x.copy(deep=True)
        generated = interpolate(x, self.dataset.categorical, np.zeros(len(x), dtype=int), 137, 2)
        self.assertEqual(len(generated), 137)
        pd.testing.assert_frame_equal(x, saved)
        for column in self.dataset.categorical:
            self.assertTrue(generated[column].isin(x[column]).all())

    def test_no_clusters_does_not_invent_synthetic_rows(self):
        x = self.dataset.features.iloc[:100]
        generated = interpolate(x, self.dataset.categorical, np.full(len(x), -1), 100, 1)
        self.assertEqual(len(generated), 0)
        labels, _, info = discover(x, 1, self.dataset.categorical)
        self.assertEqual(info["noise_count"], int((labels < 0).sum()))
        self.assertEqual(len(info["cluster_counts"]), len(set(labels) - {-1}))

    def test_sampling_does_not_change_training_rng(self):
        x = self.dataset.features.iloc[:100]
        model = MixedDiffusion(steps=5, batch_size=32).initialize(x, self.dataset.categorical).train_until(1)
        before = torch.get_rng_state().clone()
        model.sample(5)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIsInstance(model.model, GaussianMultinomialDiffusion)

    def test_both_diffusion_backbones_have_valid_decoding(self):
        x = self.dataset.features.iloc[:100]
        for backbone in ["mlp", "transformer"]:
            model = (
                MixedDiffusion(backbone, steps=5, batch_size=32)
                .initialize(x, self.dataset.categorical)
                .train_until(1)
            )
            generated = model.sample(17)
            self.assertEqual(generated.shape, (17, 23))
            for column in self.dataset.categorical:
                self.assertTrue(generated[column].isin(x[column]).all())
            for column in model.schema.num:
                self.assertTrue(generated[column].between(x[column].min(), x[column].max()).all())


if __name__ == "__main__":
    unittest.main()
