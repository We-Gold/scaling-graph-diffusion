"""
Discrete Diffusion Components for Sparse Masked Graphs.

Implements proper discrete diffusion with x_0 marginalization based on MG-Diff,
adapted for sparse graph representation with MASK and PAD tokens.

Key differences from simple CE loss:
1. Forward process q(x_t|x_0) uses proper transition matrix (not just Bernoulli masking)
2. Loss is KL[q(x_{t-1}|x_t, x_0_true) || p_θ(x_{t-1}|x_t)] via marginalization
3. p_θ(x_{t-1}|x_t) = Σ_{x_0'} q(x_{t-1}|x_t, x_0') · p_θ(x_0'|x_t)

References:
- MG-Diff: Torshin et al., "Molecular Graph Diffusion"
- D3PM: Austin et al., "Structured Denoising Diffusion Models in Discrete State Spaces"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


def log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(a)) in numerically stable way."""
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(exp(a) + exp(b))."""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from tensor `a` at indices `t` and broadcast to shape `x_shape`.

    Args:
        a: (T,) tensor of values
        t: (B,) tensor of timestep indices
        x_shape: target shape to broadcast to
    Returns:
        Extracted values broadcast to x_shape
    """
    b = t.shape[0]
    out = a[t]
    while len(out.shape) < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def index_to_log_onehot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer indices to log one-hot vectors.

    Args:
        x: (*) tensor of class indices
        num_classes: number of classes
    Returns:
        (*, num_classes) tensor of log probabilities
    """
    assert x.max().item() < num_classes, f"Index {x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x.long(), num_classes).float()
    # Move class dimension to position 1: (B, C, *spatial)
    if len(x.shape) == 1:  # (B,) -> (B, C)
        permute_order = (0, -1)
    elif len(x.shape) == 2:  # (B, N) -> (B, C, N)
        permute_order = (0, -1, 1)
    elif len(x.shape) == 3:  # (B, M, 3) for edges -> not typical for this function
        # For edges we handle each component separately
        permute_order = (0, -1) + tuple(range(1, len(x.shape)))
    else:
        permute_order = (0, -1) + tuple(range(1, len(x.shape)))

    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.clamp(min=1e-30))
    return log_x


class DiscreteDiffusionSchedule:
    """
    Discrete diffusion noise schedule with absorbing MASK token.

    Based on MG-Diff's alpha_schedule but adapted for sparse graphs where:
    - Class 0 is MASK token
    - Classes 1+ are real values
    - Forward process gradually absorbs into MASK
    - Transition: at (self-absorption), bt (uniform scatter), ct (mask absorption)
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        num_classes: int = 10,
        att_1: float = 0.99999,
        att_T: float = 0.000001,
        ctt_1: float = 0.000001,
        ctt_T: float = 0.99999,
    ):
        """
        Args:
            num_timesteps: T, number of diffusion steps
            num_classes: including MASK token at index 0
            att_1: cumulative self-absorption at t=1 (high = less noise)
            att_T: cumulative self-absorption at t=T (low = more noise)
            ctt_1: cumulative mask absorption at t=1 (low = less masking)
            ctt_T: cumulative mask absorption at t=T (high = more masking)
        """
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.N = num_classes - 1  # Number of non-mask classes

        # Compute schedule
        at, bt, ct, att, btt, ctt = self._alpha_schedule(
            num_timesteps, self.N, att_1, att_T, ctt_1, ctt_T
        )

        # Store in log space for numerical stability
        self.log_at = torch.log(at)  # (T,)
        self.log_bt = torch.log(bt)
        self.log_ct = torch.log(ct)
        self.log_cumprod_at = torch.log(att)
        self.log_cumprod_bt = torch.log(btt)
        self.log_cumprod_ct = torch.log(ctt)
        self.log_1_min_ct = log_1_min_a(self.log_ct)
        self.log_1_min_cumprod_ct = log_1_min_a(self.log_cumprod_ct)

    def _alpha_schedule(
        self, time_step: int, N: int, att_1: float, att_T: float, ctt_1: float, ctt_T: float
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute transition matrix parameters following MG-Diff.

        Transition matrix Q_t for going from x_{t-1} to x_t:
        - Q_t[0, 0] = 1 - ct (MASK stays MASK with high prob)
        - Q_t[0, i] = ct (MASK -> any other class with uniform prob)
        - Q_t[i, 0] = bt (any class -> MASK with small prob)
        - Q_t[i, i] = at (self-absorption for non-MASK)
        - Q_t[i, j] = bt for i != j, both non-MASK (uniform scatter)

        Where at + (N-1)*bt + ct ≈ 1 (rows sum to 1)
        """
        # Cumulative self-absorption schedule (linear interpolation)
        att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
        att = np.concatenate(([1], att))  # Prepend 1 for t=0
        at = att[1:] / att[:-1]  # Per-step self-absorption

        # Cumulative mask absorption schedule
        ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
        ctt = np.concatenate(([0], ctt))  # Prepend 0 for t=0
        one_minus_ctt = 1 - ctt
        one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
        ct = 1 - one_minus_ct  # Per-step mask absorption

        # Uniform scatter (residual probability)
        bt = (1 - at - ct) / N

        # Shift cumulative products for proper indexing
        # att and ctt are currently (T+1,), we want (T,) for timesteps 0..T-1
        att = att[1:]  # Remove t=0, keep t=1..T  giving us (T,)
        ctt = ctt[1:]  # Remove t=0, keep t=1..T
        btt = (1 - att - ctt) / N

        # Convert to tensors (use float32 for compatibility with PyTorch models)
        at = torch.tensor(at, dtype=torch.float32)
        bt = torch.tensor(bt, dtype=torch.float32)
        ct = torch.tensor(ct, dtype=torch.float32)
        att = torch.tensor(att, dtype=torch.float32)
        btt = torch.tensor(btt, dtype=torch.float32)
        ctt = torch.tensor(ctt, dtype=torch.float32)

        return at, bt, ct, att, btt, ctt

    def to(self, device: torch.device):
        """Move all tensors to device."""
        self.log_at = self.log_at.to(device)
        self.log_bt = self.log_bt.to(device)
        self.log_ct = self.log_ct.to(device)
        self.log_cumprod_at = self.log_cumprod_at.to(device)
        self.log_cumprod_bt = self.log_cumprod_bt.to(device)
        self.log_cumprod_ct = self.log_cumprod_ct.to(device)
        self.log_1_min_ct = self.log_1_min_ct.to(device)
        self.log_1_min_cumprod_ct = self.log_1_min_cumprod_ct.to(device)
        return self


def q_pred(
    log_x_start: torch.Tensor,
    t: torch.Tensor,
    schedule: DiscreteDiffusionSchedule
) -> torch.Tensor:
    """
    Compute q(x_t | x_0) - forward diffusion distribution.

    Args:
        log_x_start: (B, num_classes, *) log one-hot of x_0
        t: (B,) timesteps
        schedule: diffusion schedule
    Returns:
        (B, num_classes, *) log probabilities of x_t
    """
    # Extract cumulative parameters
    log_cumprod_at = extract(schedule.log_cumprod_at, t, log_x_start.shape)
    log_cumprod_bt = extract(schedule.log_cumprod_bt, t, log_x_start.shape)
    log_cumprod_ct = extract(schedule.log_cumprod_ct, t, log_x_start.shape)
    log_1_min_cumprod_ct = extract(schedule.log_1_min_cumprod_ct, t, log_x_start.shape)

    # Transition formula (MG-Diff Eq. 22):
    # q(x_t = MASK | x_0) = (1 - ct~) * 1_{x_0 = MASK} + ct~
    # q(x_t = i | x_0) = at~ * 1_{x_0 = i} + bt~  for i != MASK

    # Split MASK (index 0) and non-MASK (indices 1+)
    log_probs_mask = log_add_exp(
        log_x_start[:, :1, ...] + log_1_min_cumprod_ct,  # x_0 is MASK
        log_cumprod_ct  # Absorbed into MASK
    )
    log_probs_non_mask = log_add_exp(
        log_x_start[:, 1:, ...] + log_cumprod_at,  # Self-absorption
        log_cumprod_bt  # Uniform scatter
    )

    log_probs = torch.cat([log_probs_mask, log_probs_non_mask], dim=1)
    return log_probs


def q_posterior(
    log_x_start: torch.Tensor,
    log_x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: DiscreteDiffusionSchedule,
    mask_token_id: int = 0
) -> torch.Tensor:
    """
    Compute q(x_{t-1} | x_t, x_0) - posterior for a SPECIFIC x_0.
    This is used in the marginalization step.

    Formula (Bayes): q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)

    Args:
        log_x_start: (B, num_classes, *) log one-hot of x_0
        log_x_t: (B, num_classes, *) log one-hot of x_t
        t: (B,) timesteps
        schedule: diffusion schedule
        mask_token_id: index of MASK token (default 0)
    Returns:
        (B, num_classes, *) log probabilities of x_{t-1}
    """
    # SPECIAL CASE: At t=0, there is no t-1, so just return x_0 prediction
    # Without this check, t-1=-1 would wrap around to the last timestep!
    if (t == 0).all():
        return log_x_start

    # Follows MG-Diff discrete_diffusion_scheduler.py (q_posterior): Bayes rule in log space,
    # with log_qt = q(x_t | x_0 = x_t) used as the normalizer.
    batch_size = log_x_start.shape[0]
    device = log_x_t.device

    # Get x_t indices from log one-hot
    x_t_indices = log_x_t.argmax(1)  # (B, *)

    # Create mask for positions that are MASK tokens in x_t
    if len(x_t_indices.shape) == 1:  # (B,) - single token per batch
        is_mask = (x_t_indices == mask_token_id).unsqueeze(1)  # (B, 1)
    else:  # (B, N) - multiple tokens
        is_mask = (x_t_indices == mask_token_id).unsqueeze(1)  # (B, 1, N)

    # Compute log q(x_t | x_t) - this is the "expected" distribution if we stay at x_t
    log_qt = q_pred(log_x_t, t, schedule)
    log_qt = log_qt[:, 1:, ...]  # Remove MASK dim

    # For MASK positions, override with ct (high prob of staying MASK)
    log_cumprod_ct = extract(schedule.log_cumprod_ct, t, log_x_start.shape)
    ct_cumprod_vector = log_cumprod_ct.expand_as(log_qt)
    log_qt = torch.where(is_mask.expand_as(log_qt), ct_cumprod_vector, log_qt)

    # Compute q(x_{t-1} | x_t) using one-step transition
    log_qt_one_timestep = q_pred_one_timestep(log_x_t, t, schedule)

    # Create zero vector for MASK class
    if len(x_t_indices.shape) == 1:
        log_zero_vector = torch.full((batch_size, 1), -70.0, device=device)
    else:
        log_zero_vector = torch.full((batch_size, 1, log_x_t.shape[-1]), -70.0, device=device)

    log_qt_one_timestep = torch.cat([log_zero_vector, log_qt_one_timestep[:, 1:, ...]], dim=1)

    # Override MASK positions with ct
    log_ct = extract(schedule.log_ct, t, log_x_start.shape)
    ct_vector = log_ct.expand_as(log_qt)

    if len(x_t_indices.shape) == 1:
        log_one_vector = torch.zeros((batch_size, 1), device=device)
        ct_vector_full = torch.cat([log_one_vector, ct_vector], dim=1)
    else:
        log_one_vector = torch.zeros((batch_size, 1, log_x_t.shape[-1]), device=device)
        ct_vector_full = torch.cat([log_one_vector, ct_vector], dim=1)

    log_qt_one_timestep = torch.where(
        is_mask.expand_as(log_qt_one_timestep),
        ct_vector_full,
        log_qt_one_timestep
    )

    # Compute posterior via Bayes: q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
    q = log_x_start[:, 1:, ...] - log_qt  # Log ratio
    q = torch.cat([log_zero_vector, q], dim=1)
    q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
    q = q - q_log_sum_exp  # Normalize

    # Combine with one-step transition
    log_posterior = q_pred(q, t - 1, schedule) + log_qt_one_timestep + q_log_sum_exp

    return torch.clamp(log_posterior, -70, 0)


def q_pred_one_timestep(
    log_x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: DiscreteDiffusionSchedule
) -> torch.Tensor:
    """
    Compute q(x_t | x_{t-1}) - one-step transiton.

    Args:
        log_x_t: (B, num_classes, *) log one-hot representing x_{t-1}
        t: (B,) timesteps
        schedule: diffusion schedule
    Returns:
        (B, num_classes, *) log probabilities of x_t
    """
    log_at = extract(schedule.log_at, t, log_x_t.shape)
    log_bt = extract(schedule.log_bt, t, log_x_t.shape)
    log_ct = extract(schedule.log_ct, t, log_x_t.shape)
    log_1_min_ct = extract(schedule.log_1_min_ct, t, log_x_t.shape)

    log_probs_mask = log_add_exp(
        log_x_t[:, :1, ...] + log_1_min_ct,
        log_ct
    )
    log_probs_non_mask = log_add_exp(
        log_x_t[:, 1:, ...] + log_at,
        log_bt
    )

    log_probs = torch.cat([log_probs_mask, log_probs_non_mask], dim=1)
    return log_probs


def compute_posterior_with_marginalization(
    log_model_pred_x0: torch.Tensor,
    log_x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: DiscreteDiffusionSchedule,
    mask_token_id: int = 0
) -> torch.Tensor:
    """
    KEY FUNCTION: Compute p_θ(x_{t-1} | x_t) by marginalizing over all possible x_0.

    Formula: p_θ(x_{t-1} | x_t) = Σ_{x_0'} q(x_{t-1} | x_t, x_0') · p_θ(x_0' | x_t)

    This is the CRITICAL step missing from the simple CE loss!

    Args:
        log_model_pred_x0: (B, num_classes, *) - model's prediction of log p(x_0 | x_t)
        log_x_t: (B, num_classes, *) - current noisy state x_t
        t: (B,) - timesteps
        schedule: diffusion schedule
        mask_token_id: index of MASK token
    Returns:
        (B, num_classes, *) - log p_θ(x_{t-1} | x_t)
    """
    # For each possible x_0', compute q(x_{t-1} | x_t, x_0')
    # This is expensive! We need to sum over all num_classes possibilities.
    # MG-Diff does this intelligently in log space using q_posterior.

    # The trick: we can compute this by passing log_model_pred_x0 as the x_0 distribution
    # into q_posterior, which will properly weight each possible x_0' by its predicted probability.

    return q_posterior(log_model_pred_x0, log_x_t, t, schedule, mask_token_id)


