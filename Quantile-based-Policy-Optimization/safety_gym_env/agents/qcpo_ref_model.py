# -*- coding: utf-8 -*-
"""
Reward-value critic plus signed monotone quantiles for the internal cost ``C=-G``.

Output contract:
    - mean_value has shape [*] and supplies the ordinary reward GAE baseline;
    - quantiles has shape [*,19] and models discounted ``C=-G`` at levels 0.05,...,0.95;
    - location/scale/shape have shape [*] and parameterize its right-tail Weibull density;
    - cost quantile 0.80 exists exactly and equals minus reward quantile 0.20 in distribution.

Parameterization decisions:
    - the first atom is unconstrained and therefore supports signed negated returns;
    - softplus increments make every later atom strictly larger than its predecessor;
    - location lies strictly below Q_0.05, ensuring fitted tail atoms are in support;
    - scale and shape use softplus+epsilon, preventing invalid density parameters;
    - Weibull heads are calibrated to detached learned atoms, while QR-TD alone trains atoms;
    - finite-difference density remains a detached diagnostic and never enters actor loss.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model import init_weights


@dataclass
class RewardValueInfo:
    """All critic outputs for one leading batch shape."""
    mean_value: torch.Tensor
    quantiles: torch.Tensor
    location: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor


class RewardQuantileModel(nn.Module):
    """
    Feed-forward heads over one externally assembled finite-history state block.

    QCPORefGPU supplies `[observation, remaining_fraction, detached actor h, detached actor c]`.
    Keeping context assembly outside this module makes the critic input auditable while preserving
    the original independent Tanh MLP and its mean/quantile/Weibull heads. The first quantile is
    signed; softplus increments make all later atoms monotone without forcing positive support.
    """

    def __init__(self, state_dim, quantile_levels=None, hidden=(512, 512), epsilon=1e-4):
        super().__init__()
        levels = quantile_levels if quantile_levels is not None else             [0.05 * i for i in range(1, 20)]
        self.register_buffer("quantile_levels", torch.tensor(levels, dtype=torch.float32))
        self.epsilon = float(epsilon)
        dims = [int(state_dim)] + [int(x) for x in hidden]
        layers = []
        for index in range(len(dims) - 1):
            layers.extend([init_weights(nn.Linear(dims[index], dims[index + 1])), nn.Tanh()])
        self.encoder = nn.Sequential(*layers)
        width = dims[-1]
        self.mean_head = init_weights(nn.Linear(width, 1), gain=1.0)
        self.base_quantile_head = init_weights(nn.Linear(width, 1), gain=1.0)
        self.increment_head = init_weights(nn.Linear(width, len(levels) - 1), gain=0.1)
        # location=q_0.05-positive_gap places every cost atom inside the shifted-Weibull support.
        self.location_gap_head = init_weights(nn.Linear(width, 1), gain=0.1)
        self.scale_head = init_weights(nn.Linear(width, 1), gain=0.1)
        self.shape_head = init_weights(nn.Linear(width, 1), gain=0.1)

    def forward(self, states):
        """states [*,D] -> signed monotone quantiles [*,19] and Weibull parameters [*]."""
        leading = states.shape[:-1]
        encoded = self.encoder(states.reshape(-1, states.shape[-1]))
        first = self.base_quantile_head(encoded)
        increments = F.softplus(self.increment_head(encoded)) + self.epsilon
        quantiles = torch.cat([first, first + torch.cumsum(increments, dim=-1)], dim=-1)
        location_gap = F.softplus(self.location_gap_head(encoded)).squeeze(-1) + self.epsilon
        # Detach the anchor exactly as the target atoms are detached in the Weibull fit. Without
        # this detach, Weibull calibration silently pushes the QR quantile head through location,
        # despite the intended separation between distributional TD and density approximation.
        location = quantiles[:, 0].detach() - location_gap
        scale = F.softplus(self.scale_head(encoded)).squeeze(-1) + self.epsilon
        shape = F.softplus(self.shape_head(encoded)).squeeze(-1) + self.epsilon
        mean_value = self.mean_head(encoded).squeeze(-1)
        return RewardValueInfo(
            mean_value=mean_value.reshape(leading),
            quantiles=quantiles.reshape(*leading, -1),
            location=location.reshape(leading),
            scale=scale.reshape(leading),
            shape=shape.reshape(leading),
        )

    def quantile_index(self, alpha):
        """Return one exact level index; both 0.20 and its mirrored 0.80 are on the grid."""
        matches = torch.nonzero(torch.isclose(
            self.quantile_levels, torch.tensor(float(alpha), device=self.quantile_levels.device)),
            as_tuple=False).flatten()
        if matches.numel() != 1:
            raise ValueError(f"alpha={alpha} is not represented exactly")
        return int(matches.item())
