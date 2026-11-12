# -*- coding: utf-8 -*-
"""Author-aligned recurrent model for the shifted-Weibull QCPO reference.

This module intentionally follows the executable model in Jung et al.'s QCPO
repository.  One shared feature stream is used by the Gaussian policy, reward
value, independent cost-quantile, and Weibull heads:

``augmented observation -> MLP(512, 512, Tanh)``
``[MLP feature, previous action, previous reward] -> LSTM(512)``
``shared feature = MLP feature + LSTM output``

The augmented observation is expected to contain the previous internal cost;
the collector owns that augmentation.  Consequently this model accepts the
same three explicit inputs as the author's model: augmented observation,
previous action, and previous reward.

Only the distribution support differs from the two-parameter official model:
the 25 quantile heads remain independent but are signed, and a learned location
is added to form a shifted Weibull.  Its anchor is the *detached* smallest
predicted atom, so Weibull fitting cannot update the quantile head through the
location path.  Shape and scale retain the author's exact transforms,
``4 * sigmoid`` and ``exp`` respectively.
"""

from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AuthorAlignedValueInfo(NamedTuple):
    """All scalar/distributional heads with the input's leading dimensions."""

    r_value: torch.Tensor
    c_dist: Optional[torch.Tensor]
    c_w_location: Optional[torch.Tensor]
    c_w_alpha: Optional[torch.Tensor]
    c_w_beta: Optional[torch.Tensor]


class AuthorAlignedRnnState(NamedTuple):
    """PyTorch LSTM state, always stored as ``[1, B, lstm_size]`` tensors."""

    h: torch.Tensor
    c: torch.Tensor


class RunningMeanStdModel(nn.Module):
    """Torch-buffer port of rlpyt's observation running statistics.

    The initial state is deliberately ``mean=0, var=1, count=0``.  In
    particular, no artificial epsilon count is used: the first update replaces
    mean and variance with that batch's population statistics, exactly as in
    the author's rlpyt dependency.
    """

    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self.shape = tuple(int(size) for size in shape)

        # Buffers follow the model across devices and are serialized in its
        # state_dict, but are never optimized by Adam.
        self.register_buffer("mean", torch.zeros(self.shape))
        self.register_buffer("var", torch.ones(self.shape))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, observation: torch.Tensor) -> None:
        """Merge all leading samples using the parallel-variance formula."""

        if tuple(observation.shape[-len(self.shape):]) != self.shape:
            raise ValueError(
                "observation trailing shape must equal RMS shape: "
                f"got {tuple(observation.shape)}, expected *{self.shape}"
            )

        # A bare observation has no explicit batch axis; every other leading
        # axis is flattened because RMS statistics are feature-wise.
        flat = observation.reshape(-1, *self.shape)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = flat.shape[0]

        if self.count.item() == 0:
            # rlpyt special-cases the first batch instead of mixing it with the
            # initialized unit variance.
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            merged_mean = self.mean + delta * batch_count / total

            # M2 is the sum of squared deviations for both populations plus
            # the correction for their different means.
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            merged_m2 = (
                m_a
                + m_b
                + delta.square() * self.count * batch_count / total
            )
            self.mean.copy_(merged_mean)
            self.var.copy_(merged_m2 / total)

        self.count.add_(batch_count)


def _infer_leading_dims(
    tensor: torch.Tensor,
    data_ndim: int,
) -> Tuple[int, int, int]:
    """Infer rlpyt's optional ``[]``, ``[B]``, or ``[T,B]`` leading axes."""

    lead_dim = tensor.dim() - int(data_ndim)
    if lead_dim not in (0, 1, 2):
        raise ValueError(
            "model inputs support zero, one, or two leading dimensions; "
            f"received tensor shape {tuple(tensor.shape)}"
        )
    if lead_dim == 2:
        time, batch = tensor.shape[:2]
    elif lead_dim == 1:
        time, batch = 1, tensor.shape[0]
    else:
        time, batch = 1, 1
    return lead_dim, int(time), int(batch)


def _restore_leading_dims(
    tensor: torch.Tensor,
    lead_dim: int,
    time: int,
    batch: int,
) -> torch.Tensor:
    """Restore a flattened ``[T*B,...]`` output to match its input layout."""

    if lead_dim == 2:
        return tensor.reshape(time, batch, *tensor.shape[1:])
    if lead_dim == 0:
        return tensor.squeeze(0)
    return tensor


class QcpoRefAuthorAlignedModel(nn.Module):
    """Shared MLP-LSTM QCPO model with signed atoms and a shifted Weibull.

    Parameters mirror the original ``QcpoModel`` where possible.  The defaults
    are the paper/repository configuration: two 512-unit Tanh layers, one
    512-unit LSTM, residual LSTM skip, 25 midpoint quantiles, state-independent
    Gaussian log standard deviation initialized to zero, and observation RMS.

    All ``Linear`` and ``LSTM`` modules intentionally retain PyTorch's default
    initialization.  No project-wide orthogonal initializer is called here.
    """

    def __init__(
        self,
        observation_shape: Union[int, Sequence[int]],
        action_size: int,
        n_quantile: int = 25,
        hidden_sizes: Sequence[int] = (512, 512),
        lstm_size: int = 512,
        lstm_skip: bool = True,
        constraint: bool = True,
        init_log_std: float = 0.0,
        normalize_observation: bool = True,
        var_clip: float = 1e-6,
        location_epsilon: float = 1e-6,
        raw_cost_support: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(observation_shape, int):
            observation_shape = (observation_shape,)
        self.observation_shape = tuple(int(size) for size in observation_shape)
        self.action_size = int(action_size)
        self.n_quantile = int(n_quantile)
        self.lstm_size = int(lstm_size)
        self.lstm_skip = bool(lstm_skip)
        self.normalize_observation = bool(normalize_observation)
        self.var_clip = float(var_clip)
        self.location_epsilon = float(location_epsilon)
        self.raw_cost_support = bool(raw_cost_support)

        if len(hidden_sizes) != 2:
            raise ValueError(
                "author-aligned body requires exactly two hidden layers; "
                f"received {tuple(hidden_sizes)}"
            )
        if self.n_quantile <= 0:
            raise ValueError("n_quantile must be positive")
        if self.location_epsilon <= 0:
            raise ValueError("location_epsilon must be positive")

        input_size = int(np.prod(self.observation_shape))
        hidden_0, hidden_1 = (int(size) for size in hidden_sizes)

        # This explicit Sequential reproduces MlpModel's two Linear/Tanh
        # stages while leaving every parameter at native PyTorch defaults.
        self.body = nn.Sequential(
            nn.Linear(input_size, hidden_0),
            nn.Tanh(),
            nn.Linear(hidden_0, hidden_1),
            nn.Tanh(),
        )

        # The author feeds the current MLP feature plus previous action and
        # previous reward to the LSTM.  Previous cost is already the final
        # coordinate of the externally augmented observation.
        lstm_input_size = hidden_1 + self.action_size + 1
        self.lstm = nn.LSTM(lstm_input_size, self.lstm_size)
        if self.lstm_skip and hidden_1 != self.lstm_size:
            raise ValueError(
                "lstm_skip requires the final MLP width to equal lstm_size: "
                f"got {hidden_1} and {self.lstm_size}"
            )

        shared_size = self.lstm_size
        self.mu = nn.Sequential(nn.Linear(shared_size, self.action_size), nn.Tanh())
        self.value = nn.Linear(shared_size, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_size,), float(init_log_std))
        )

        # Midpoint levels match the official 25-atom grid:
        # 0.02, 0.06, ..., 0.98 when n_quantile=25.
        tau = (
            torch.arange(self.n_quantile, dtype=torch.float32) / self.n_quantile
            + 0.5 / self.n_quantile
        )
        # The original keeps these as non-parameter attributes.  Non-persistent
        # buffers preserve that checkpoint surface while still moving them with
        # `.to(device)`, which the historical plain attributes failed to do.
        self.register_buffer("tau", tau, persistent=False)
        # Keep the released expression literally as ``-log(1-tau)``.  ``log1p`` is analytically
        # equivalent but differs by one float32 ULP at 9/25 levels and therefore is not executable
        # parity for fixed-weight differential tests.
        self.register_buffer("c_tau", -torch.log(1.0 - tau), persistent=False)

        if constraint:
            # Unlike the official positive-cost head, this Linear output has
            # no exp/softplus transform: each atom is signed and independent.
            self.constraint = nn.Linear(shared_size, self.n_quantile)

            # Retain the author's exact shape and scale parameterizations.
            # These are constructed before the new location head so a fixed
            # seed gives all original heads their historical initialization.
            self.w_alpha = nn.Linear(shared_size, 1)
            self.w_beta = nn.Linear(shared_size, 1)

            # Location is anchored below the detached minimum atom.  Creating
            # this one genuinely new head last avoids perturbing common-head
            # RNG initialization relative to the author's model.
            self.w_location_gap = None if self.raw_cost_support else nn.Linear(shared_size, 1)
        else:
            self.constraint = None
            self.w_location_gap = None
            self.w_alpha = None
            self.w_beta = None

        self.obs_rms = (
            RunningMeanStdModel(self.observation_shape)
            if self.normalize_observation
            else None
        )

    def forward(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        init_rnn_state: Optional[
            Union[AuthorAlignedRnnState, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        """Evaluate inputs with optional leading ``[]``, ``[B]``, or ``[T,B]``.

        Returns ``(mu, log_std, value_info, next_rnn_state)``.  For a time-major
        input ``[T,B,D]``, policy/value heads have shapes ``[T,B,...]`` and the
        recurrent state has shape ``[1,B,lstm_size]``.
        """

        lead_dim, time, batch = _infer_leading_dims(
            observation, len(self.observation_shape)
        )
        expected_trailing = self.observation_shape
        if tuple(observation.shape[-len(expected_trailing):]) != expected_trailing:
            raise ValueError(
                "observation trailing shape mismatch: "
                f"got {tuple(observation.shape)}, expected *{expected_trailing}"
            )

        if self.normalize_observation:
            # The variance floor and final clip are exactly the official model
            # operations; the floor is not added to the RMS count/update.
            obs_variance = torch.clamp(self.obs_rms.var, min=self.var_clip)
            observation = torch.clamp(
                (observation - self.obs_rms.mean) / obs_variance.sqrt(),
                min=-10.0,
                max=10.0,
            )

        flat_observation = observation.reshape(time * batch, -1)
        mlp_feature = self.body(flat_observation)

        # `.reshape(T,B,-1)` intentionally mirrors rlpyt for all supported
        # leading layouts, including scalar previous rewards.
        recurrent_input = torch.cat(
            (
                mlp_feature.reshape(time, batch, -1),
                prev_action.reshape(time, batch, -1),
                prev_reward.reshape(time, batch, -1),
            ),
            dim=-1,
        )
        initial_state = (
            None if init_rnn_state is None else tuple(init_rnn_state)
        )
        recurrent_output, (next_h, next_c) = self.lstm(
            recurrent_input, initial_state
        )
        recurrent_feature = recurrent_output.reshape(time * batch, -1)

        # The residual is a defining part of the official architecture: every
        # downstream head sees both instantaneous MLP and recurrent features.
        shared_feature = (
            mlp_feature + recurrent_feature
            if self.lstm_skip
            else recurrent_feature
        )

        mu = self.mu(shared_feature)
        log_std = self.log_std.repeat(time * batch, 1)
        r_value = self.value(shared_feature).squeeze(-1)

        if self.constraint is None:
            value_info = AuthorAlignedValueInfo(
                r_value=_restore_leading_dims(
                    r_value, lead_dim, time, batch
                ),
                c_dist=None,
                c_w_location=None,
                c_w_alpha=None,
                c_w_beta=None,
            )
        else:
            # Signed, independent atoms: sorting is deliberately deferred to
            # target/GAE/tail utilities, just as requested by the frozen port.
            c_dist = self.constraint(shared_feature)
            # 原始非负 cost 使用官方 exp quantile head 和零 location Weibull。
            # signed-return 历史实现仍保留；raw-cost 子类在构造后显式启用。
            if getattr(self, "raw_cost_support", False):
                c_dist = torch.exp(c_dist)
            if self.raw_cost_support:
                c_w_location = torch.zeros_like(c_dist[..., 0])
            else:
                minimum_atom = c_dist.min(dim=-1).values.detach()
                location_gap = F.softplus(self.w_location_gap(shared_feature).squeeze(-1))
                c_w_location = minimum_atom - location_gap - self.location_epsilon
            c_w_alpha = 4.0 * torch.sigmoid(
                self.w_alpha(shared_feature).squeeze(-1)
            )
            c_w_beta = torch.exp(
                self.w_beta(shared_feature).squeeze(-1)
            )

            value_info = AuthorAlignedValueInfo(
                r_value=_restore_leading_dims(
                    r_value, lead_dim, time, batch
                ),
                c_dist=_restore_leading_dims(
                    c_dist, lead_dim, time, batch
                ),
                c_w_location=_restore_leading_dims(
                    c_w_location, lead_dim, time, batch
                ),
                c_w_alpha=_restore_leading_dims(
                    c_w_alpha, lead_dim, time, batch
                ),
                c_w_beta=_restore_leading_dims(
                    c_w_beta, lead_dim, time, batch
                ),
            )

        mu = _restore_leading_dims(mu, lead_dim, time, batch)
        log_std = _restore_leading_dims(log_std, lead_dim, time, batch)
        next_state = AuthorAlignedRnnState(h=next_h, c=next_c)
        return mu, log_std, value_info, next_state

    @torch.no_grad()
    def update_obs_rms(self, observation: torch.Tensor) -> None:
        """Update augmented-observation statistics; no-op if RMS is disabled."""

        if self.obs_rms is not None:
            self.obs_rms.update(observation)


# Short aliases make the module convenient for adapters while the long class
# names keep checkpoints and diagnostics unambiguous.
ValueInfo = AuthorAlignedValueInfo
RnnState = AuthorAlignedRnnState
QcpoAuthorAlignedModel = QcpoRefAuthorAlignedModel


__all__ = [
    "AuthorAlignedRnnState",
    "AuthorAlignedValueInfo",
    "QcpoAuthorAlignedModel",
    "QcpoRefAuthorAlignedModel",
    "RnnState",
    "RunningMeanStdModel",
    "ValueInfo",
]
