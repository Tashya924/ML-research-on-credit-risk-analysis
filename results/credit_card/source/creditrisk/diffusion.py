"""Official TabDDPM loss engine with explicit project backbone/sampling adapters."""

import contextlib
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from vendor.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from vendor.tab_ddpm.modules import MLPDiffusion, timestep_embedding

from .data import Schema


class FeatureDenoiser(nn.Module):
    """One token per original feature, not one token for the entire row."""

    def __init__(self, numeric_count, cardinalities, clusters, width=32):
        super().__init__()
        self.numeric_count = numeric_count
        self.cardinalities = cardinalities
        self.numeric_weight = nn.Parameter(torch.randn(numeric_count, width) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(numeric_count, width))
        self.category_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(k, width) * 0.02) for k in cardinalities]
        )
        self.position = nn.Parameter(torch.randn(numeric_count + len(cardinalities), width) * 0.02)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.condition = nn.Embedding(clusters, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, 2 * width, dropout=0, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.numeric_head = nn.Linear(width, 1)
        self.category_heads = nn.ModuleList([nn.Linear(width, k) for k in cardinalities])

    def forward(self, x, timesteps, y=None):
        numbers = x[:, : self.numeric_count]
        categorical = x[:, self.numeric_count :].split(self.cardinalities, dim=1)
        category_tokens = torch.stack(
            [
                values.exp() @ weights
                for values, weights in zip(categorical, self.category_embeddings, strict=False)
            ],
            dim=1,
        )
        numeric_tokens = numbers.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        tokens = torch.cat([numeric_tokens, category_tokens], dim=1)
        condition = self.time(timestep_embedding(timesteps, self.position.shape[1]))
        if y is not None:
            condition = condition + self.condition(y)
        hidden = self.encoder(tokens + self.position + condition.unsqueeze(1))
        numeric_output = self.numeric_head(hidden[:, : self.numeric_count]).squeeze(-1)
        category_output = [
            head(hidden[:, self.numeric_count + i]) for i, head in enumerate(self.category_heads)
        ]
        return torch.cat([numeric_output, *category_output], dim=1)


class BoundedDiffusion(GaussianMultinomialDiffusion):
    """Project sampling adaptation; official variational training loss unchanged."""

    def gaussian_p_mean_variance(self, model_output, x, t, **kwargs):
        result = super().gaussian_p_mean_variance(model_output, x, t, **kwargs)
        clean = result["pred_xstart"].clamp(self.clean_lower, self.clean_upper)
        result["mean"] = self.gaussian_q_posterior_mean_variance(clean, x, t)[0]
        result["pred_xstart"] = clean
        return result


class MixedDiffusion:
    """Incremental fitting permits validation-based checkpoint selection without refits."""

    def __init__(self, backbone="mlp", steps=100, batch_size=512, seed=42):
        if backbone not in {"mlp", "transformer"} or steps < 2 or batch_size < 1:
            raise ValueError("Invalid diffusion configuration")
        self.backbone = backbone
        self.steps = steps
        self.batch_size = batch_size
        self.seed = seed

    def initialize(self, x, categorical, labels=None):
        torch.manual_seed(self.seed)
        self.schema = Schema().fit(x, self.seed, categorical)
        numbers, categories = self.schema.encode(x)
        self.training = torch.tensor(np.concatenate([numbers, categories], axis=1), dtype=torch.float32)
        labels = np.zeros(len(x), dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
        if len(labels) != len(x) or labels.min() < 0:
            raise ValueError("Cluster labels must match generator training records")
        self.labels = torch.tensor(labels)
        dimension = numbers.shape[1] + sum(self.schema.cards)
        clusters = int(labels.max()) + 1
        if self.backbone == "mlp":
            denoiser = MLPDiffusion(
                dimension, clusters, True, dict(d_layers=[128, 128], dropout=0.0), dim_t=128
            )
        else:
            denoiser = FeatureDenoiser(numbers.shape[1], self.schema.cards, clusters)
        self.model = BoundedDiffusion(
            np.asarray(self.schema.cards),
            numbers.shape[1],
            denoiser,
            num_timesteps=self.steps,
            multinomial_loss_type="vb_stochastic",
            scheduler="cosine",
        )
        self.model.clean_lower = torch.tensor(
            self.schema.qt.transform(self.schema.bounds[0][None, :])[0], dtype=torch.float32
        )
        self.model.clean_upper = torch.tensor(
            self.schema.qt.transform(self.schema.bounds[1][None, :])[0], dtype=torch.float32
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.history = []
        self.batch_history = []
        self.epoch = 0
        return self

    def train_until(self, epochs):
        self.model.train()
        while self.epoch < epochs:
            # Epoch-specific RNG decouples optimization from validation sampling.
            torch.manual_seed(self.seed + self.epoch * 104729)
            total = numeric_total = categorical_total = 0.0
            batch_number = 0
            for indices in torch.randperm(len(self.training)).split(self.batch_size):
                categorical_loss, numeric_loss = self.model.mixed_loss(
                    self.training[indices], {"y": self.labels[indices]}
                )
                loss = categorical_loss + numeric_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diffusion training loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                batch_number += 1
                value, numeric_value, categorical_value = (
                    loss.item(),
                    numeric_loss.item(),
                    categorical_loss.item(),
                )
                total += value * len(indices)
                numeric_total += numeric_value * len(indices)
                categorical_total += categorical_value * len(indices)
                self.batch_history.append(
                    dict(
                        epoch=self.epoch + 1,
                        batch=batch_number,
                        examples=len(indices),
                        training_loss=value,
                        numeric_mse=numeric_value,
                        categorical_vb=categorical_value,
                    )
                )
            self.epoch += 1
            self.history.append(
                dict(
                    epoch=self.epoch,
                    training_loss=total / len(self.training),
                    numeric_mse=numeric_total / len(self.training),
                    categorical_vb=categorical_total / len(self.training),
                )
            )
        return self

    def state(self):
        return copy.deepcopy(self.model.state_dict())

    @torch.no_grad()
    def sample(self, count, probabilities=None):
        if count < 1:
            raise ValueError("Positive sample count required")
        probabilities = np.asarray([1.0] if probabilities is None else probabilities)
        if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
            raise ValueError("Invalid cluster probabilities")
        self.model.eval()
        frames = []
        # Sampling must not reset the training process's RNG state.
        with torch.random.fork_rng(), open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            torch.manual_seed(self.seed + 1000)
            for start in range(0, count, self.batch_size):
                size = min(self.batch_size, count - start)
                values, _ = self.model.sample(size, torch.tensor(probabilities, dtype=torch.float32))
                array = values.numpy()
                numeric_count = len(self.schema.num)
                frames.append(
                    self.schema.decode(array[:, :numeric_count], array[:, numeric_count:].astype(np.int64))
                )
        return pd.concat(frames, ignore_index=True)
