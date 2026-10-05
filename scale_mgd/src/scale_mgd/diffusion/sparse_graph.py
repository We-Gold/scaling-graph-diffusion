"""
Sparse Graph Discrete Diffusion - Adapter for sparse graph representation.

Two modes:
1. atomic_edges=True (DEFAULT): Treats edges as atomic (u,v,t) tuples
   - Entire edge is masked or unmasked together
   - 2 schedules: nodes (X) and edges (E)
   - More semantically coherent for sparse graphs

2. atomic_edges=False: Independent masking of edge components
   - Each of (u, v, t) masked independently
   - 4 schedules: X, Eu, Ev, Et
   - Useful for ablation studies
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .discrete import (
    DiscreteDiffusionSchedule,
    q_pred,
    q_posterior,
    compute_posterior_with_marginalization,
    index_to_log_onehot,
)


class SparseGraphDiffusionSchedule:
    """
    Multi-modal discrete diffusion schedule for sparse graphs.

    Default behavior (atomic_edges=True):
    - Nodes: X with num_node_types classes
    - Edges: (u, v, t) tuples masked together as atomic units

    Alternative (atomic_edges=False):
    - Nodes: X
    - Edge sources: u (independent)
    - Edge targets: v (independent)
    - Edge types: t (independent)
    """

    def __init__(
        self,
        num_timesteps: int,
        num_node_types: int,
        num_edge_types: int,
        max_nodes: int,
        atomic_edges: bool = True,
        att_1: float = 0.99999,
        att_T: float = 0.000001,
        ctt_1: float = 0.000001,
        ctt_T: float = 0.99999,
    ):
        """
        Args:
            num_timesteps: T, number of diffusion steps
            num_node_types: number of node type classes (includes MASK, PAD)
            num_edge_types: number of edge type classes (includes MASK, NO_BOND)
            max_nodes: maximum number of nodes (e.g., 38 for ZINC)
            atomic_edges: if True (default), mask entire (u,v,t) tuples together
            att_1, att_T, ctt_1, ctt_T: schedule parameters
        """
        self.num_timesteps = num_timesteps
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.max_nodes = max_nodes
        self.atomic_edges = atomic_edges

        # Schedule for node types: X (always needed)
        self.schedule_X = DiscreteDiffusionSchedule(
            num_timesteps=num_timesteps,
            num_classes=num_node_types,
            att_1=att_1,
            att_T=att_T,
            ctt_1=ctt_1,
            ctt_T=ctt_T,
        )

        if atomic_edges:
            # Single schedule for edge tuples
            # We use a binary schedule: edge is masked (0) or present (1)
            # When present, we noise the individual components separately
            # but the decision to mask the edge is atomic
            self.schedule_E = DiscreteDiffusionSchedule(
                num_timesteps=num_timesteps,
                num_classes=2,  # Binary: masked or present
                att_1=att_1,
                att_T=att_T,
                ctt_1=ctt_1,
                ctt_T=ctt_T,
            )

            # Schedules for component values when edge is present
            # These are used for noising the actual values, not for masking
            num_node_pointer_classes = max_nodes + 1
            self.schedule_Eu_vals = DiscreteDiffusionSchedule(
                num_timesteps=num_timesteps,
                num_classes=num_node_pointer_classes,
                att_1=att_1,
                att_T=att_T,
                ctt_1=ctt_1,
                ctt_T=ctt_T,
            )
            self.schedule_Ev_vals = DiscreteDiffusionSchedule(
                num_timesteps=num_timesteps,
                num_classes=num_node_pointer_classes,
                att_1=att_1,
                att_T=att_T,
                ctt_1=ctt_1,
                ctt_T=ctt_T,
            )
            self.schedule_Et_vals = DiscreteDiffusionSchedule(
                num_timesteps=num_timesteps,
                num_classes=num_edge_types,
                att_1=att_1,
                att_T=att_T,
                ctt_1=ctt_1,
                ctt_T=ctt_T,
            )
        else:
            # Separate schedules for each edge component (old behavior)
            num_node_pointer_classes = max_nodes + 1

            self.schedule_Eu = DiscreteDiffusionSchedule(
                num_timesteps=num_timesteps,
                num_classes=num_node_pointer_classes,
                att_1=att_1,
                att_T=att_T,
                ctt_1=ctt_1,
                ctt_T=ctt_T,
            )

            self.schedule_Ev = DiscreteDiffusionSchedule(
                num_timesteps=num_timesteps,
                num_classes=num_node_pointer_classes,
                att_1=att_1,
                att_T=att_T,
                ctt_1=ctt_1,
                ctt_T=ctt_T,
            )

            self.schedule_Et = DiscreteDiffusionSchedule(
                num_timesteps=num_timesteps,
                num_classes=num_edge_types,
                att_1=att_1,
                att_T=att_T,
                ctt_1=ctt_1,
                ctt_T=ctt_T,
            )

    def to(self, device: torch.device):
        """Move all schedules to device."""
        self.schedule_X.to(device)

        if self.atomic_edges:
            self.schedule_E.to(device)
            self.schedule_Eu_vals.to(device)
            self.schedule_Ev_vals.to(device)
            self.schedule_Et_vals.to(device)
        else:
            self.schedule_Eu.to(device)
            self.schedule_Ev.to(device)
            self.schedule_Et.to(device)

        return self


def q_sample_sparse_graph(
    X: torch.Tensor,
    E: torch.Tensor,
    t: torch.Tensor,
    schedules: SparseGraphDiffusionSchedule,
    mask_token_uv: Optional[int] = None,
    mask_token_t: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from forward process q(x_t | x_0) for sparse graphs.

    Args:
        X: (B, N) clean node types
        E: (B, M, 3) clean edges [u, v, type]
        t: (B,) timesteps
        schedules: multi-modal schedules
        mask_token_uv: MASK token for u/v (defaults to max_nodes)
        mask_token_t: MASK token for edge types
    Returns:
        X_t: (B, N) noisy node types
        E_t: (B, M, 3) noisy edges
    """
    B, N = X.shape
    M = E.shape[1]
    device = X.device

    if mask_token_uv is None:
        mask_token_uv = schedules.max_nodes

    # Sample X_t (same for both modes)
    log_X_0 = index_to_log_onehot(X, schedules.num_node_types)
    log_X_t = q_pred(log_X_0, t, schedules.schedule_X)
    X_t = torch.multinomial(log_X_t.exp().transpose(1, 2).reshape(-1, schedules.num_node_types), 1)
    X_t = X_t.reshape(B, N)

    # CRITICAL: Remap u/v to canonical ordering (MASK at index 0)
    # Input: u, v ∈ {0, 1, ..., max_nodes-1, max_nodes} where max_nodes=MASK
    # Output: u, v ∈ {0=MASK, 1, 2, ..., max_nodes} where 0=MASK
    Eu, Ev, Et = E[:, :, 0], E[:, :, 1], E[:, :, 2]

    # Remap: if value==mask_token_uv -> 0, else value+1
    Eu_remapped = torch.where(Eu == mask_token_uv, 0, Eu + 1)
    Ev_remapped = torch.where(Ev == mask_token_uv, 0, Ev + 1)

    if schedules.atomic_edges:
        # ATOMIC MODE: Decide which edges to mask entirely
        # Create binary indicators: 0 = masked, 1 = present
        E_indicators = torch.ones(B, M, dtype=torch.long, device=device)

        log_E_0 = index_to_log_onehot(E_indicators, 2)  # Binary
        log_E_t = q_pred(log_E_0, t, schedules.schedule_E)
        E_t_indicators = torch.multinomial(log_E_t.exp().transpose(1, 2).reshape(-1, 2), 1)
        E_t_indicators = E_t_indicators.reshape(B, M)

        # For edges that are present (indicator=1), noise the components
        # For edges that are masked (indicator=0), set all to MASK
        # NOTE: Now using remapped indices where MASK=0
        log_Eu_0 = index_to_log_onehot(Eu_remapped, schedules.max_nodes + 1)
        log_Eu_t = q_pred(log_Eu_0, t, schedules.schedule_Eu_vals)
        Eu_t = torch.multinomial(log_Eu_t.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1)
        Eu_t = Eu_t.reshape(B, M)

        log_Ev_0 = index_to_log_onehot(Ev_remapped, schedules.max_nodes + 1)
        log_Ev_t = q_pred(log_Ev_0, t, schedules.schedule_Ev_vals)
        Ev_t = torch.multinomial(log_Ev_t.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1)
        Ev_t = Ev_t.reshape(B, M)

        log_Et_0 = index_to_log_onehot(Et, schedules.num_edge_types)
        log_Et_t = q_pred(log_Et_0, t, schedules.schedule_Et_vals)
        Et_t = torch.multinomial(log_Et_t.exp().transpose(1, 2).reshape(-1, schedules.num_edge_types), 1)
        Et_t = Et_t.reshape(B, M)

        # Apply atomic masking: if indicator=0, mask all components (MASK=0 now)
        is_masked = (E_t_indicators == 0)
        Eu_t = torch.where(is_masked, torch.zeros_like(Eu_t), Eu_t)
        Ev_t = torch.where(is_masked, torch.zeros_like(Ev_t), Ev_t)
        Et_t = torch.where(is_masked, mask_token_t, Et_t)

    else:
        # INDEPENDENT MODE: Each component masked separately (old behavior)
        # NOTE: Now using remapped indices where MASK=0
        log_Eu_0 = index_to_log_onehot(Eu_remapped, schedules.max_nodes + 1)
        log_Eu_t = q_pred(log_Eu_0, t, schedules.schedule_Eu)
        Eu_t = torch.multinomial(log_Eu_t.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1)
        Eu_t = Eu_t.reshape(B, M)

        # Sample v_t
        log_Ev_0 = index_to_log_onehot(Ev_remapped, schedules.max_nodes + 1)
        log_Ev_t = q_pred(log_Ev_0, t, schedules.schedule_Ev)
        Ev_t = torch.multinomial(log_Ev_t.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1)
        Ev_t = Ev_t.reshape(B, M)

        # Sample t_t (edge types)
        log_Et_0 = index_to_log_onehot(Et, schedules.num_edge_types)
        log_Et_t = q_pred(log_Et_0, t, schedules.schedule_Et)
        Et_t = torch.multinomial(log_Et_t.exp().transpose(1, 2).reshape(-1, schedules.num_edge_types), 1)
        Et_t = Et_t.reshape(B, M)

    # CRITICAL: Reverse remap back to original coordinate system
    # Output: {0=MASK, 1, 2, ..., max_nodes} -> {0, 1, ..., max_nodes-1, max_nodes=MASK}
    Eu_t = torch.where(Eu_t == 0, mask_token_uv, Eu_t - 1)
    Ev_t = torch.where(Ev_t == 0, mask_token_uv, Ev_t - 1)

    E_t = torch.stack([Eu_t, Ev_t, Et_t], dim=-1)

    return X_t, E_t


class SparseGraphDiffusionLoss:
    """
    VLB loss for sparse graph discrete diffusion.

    Handles both atomic and independent edge masking modes.
    """

    def __init__(
        self,
        schedules: SparseGraphDiffusionSchedule,
        auxiliary_loss_weight: float = 1e-4,
        adaptive_auxiliary_loss: bool = True,
        mask_weight: Tuple[float, float] = (1.0, 1.0),
        mask_token_X: int = 0,
        mask_token_uv: Optional[int] = None,
        mask_token_t: int = 0,
    ):
        """
        Args:
            schedules: multi-modal schedules
            auxiliary_loss_weight: weight for x_0 prediction loss
            adaptive_auxiliary_loss: whether to scale aux loss by (1 - t/T)
            mask_weight: (weight_for_mask, weight_for_non_mask)
            mask_token_X: MASK token for nodes (default 0)
            mask_token_uv: MASK token for u/v (default max_nodes)
            mask_token_t: MASK token for edge types (default 0)
        """
        self.schedules = schedules
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        self.mask_token_X = mask_token_X
        self.mask_token_uv = mask_token_uv if mask_token_uv is not None else schedules.max_nodes
        self.mask_token_t = mask_token_t

    def _compute_modality_loss(
        self,
        log_model_pred_x0: torch.Tensor,
        x_true: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pt: torch.Tensor,
        schedule: DiscreteDiffusionSchedule,
        mask_token: int,
    ) -> torch.Tensor:
        """
        Compute VLB loss for a single modality.

        Args:
            log_model_pred_x0: (B, num_classes, *) model's log p(x_0 | x_t)
            x_true: (B, *) ground truth x_0
            x_t: (B, *) noisy x_t
            t: (B,) timesteps
            pt: (B,) timestep probability
            schedule: diffusion schedule for this modality
            mask_token: MASK token ID for this modality
        Returns:
            (B,) loss per sample
        """
        num_classes = log_model_pred_x0.shape[1]

        # Convert to log one-hot
        log_x_true = index_to_log_onehot(x_true, num_classes)
        log_x_t = index_to_log_onehot(x_t, num_classes)

        # Compute model posterior via marginalization
        log_model_prob = compute_posterior_with_marginalization(
            log_model_pred_x0, log_x_t, t, schedule, mask_token
        )

        # Compute true posterior
        log_true_prob = q_posterior(log_x_true, log_x_t, t, schedule, mask_token)

        # KL divergence
        kl = (log_true_prob.exp() * (log_true_prob - log_model_prob)).sum(dim=1)

        # Apply mask weighting
        is_mask = (x_t == mask_token).float()
        weight = is_mask * self.mask_weight[0] + (1 - is_mask) * self.mask_weight[1]
        kl = kl * weight

        # Decoder NLL for t=0
        decoder_nll = -(log_x_true.exp() * log_model_prob).sum(dim=1)

        # Use KL for t>0, decoder NLL for t=0
        mask_t0 = (t == 0).float()
        if len(mask_t0.shape) < len(kl.shape):
            mask_t0 = mask_t0.unsqueeze(-1)
        kl_loss = mask_t0 * decoder_nll + (1 - mask_t0) * kl

        # Main VLB with importance weighting
        loss = kl_loss / pt.unsqueeze(-1)

        # Auxiliary loss
        if self.auxiliary_loss_weight > 0:
            kl_aux = (log_x_true.exp() * (log_x_true - log_model_pred_x0)).sum(dim=1)
            kl_aux = kl_aux * weight
            kl_aux_loss = mask_t0 * decoder_nll + (1 - mask_t0) * kl_aux

            if self.adaptive_auxiliary_loss:
                aux_weight = (1 - t.float() / schedule.num_timesteps) + 1.0
                if len(aux_weight.shape) < len(kl_aux_loss.shape):
                    aux_weight = aux_weight.unsqueeze(-1)
            else:
                aux_weight = 1.0

            loss = loss + aux_weight * self.auxiliary_loss_weight * kl_aux_loss / pt.unsqueeze(-1)

        # Sum over spatial dimensions
        return loss.sum(dim=tuple(range(1, len(loss.shape))))

    def compute_loss(
        self,
        logits_X: torch.Tensor,
        logits_Eu: torch.Tensor,
        logits_Ev: torch.Tensor,
        logits_Et: torch.Tensor,
        X_true: torch.Tensor,
        E_true: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        t: torch.Tensor,
        pt: torch.Tensor,
        mask_padding: bool = True,
        num_real_nodes: Optional[torch.Tensor] = None,
        E_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total VLB loss for sparse graph.

        Args:
            logits_X: (B, num_node_types, N) model output for nodes
            logits_Eu: (B, max_nodes+1, M) model output for edge sources
            logits_Ev: (B, max_nodes+1, M) model output for edge targets
            logits_Et: (B, num_edge_types, M) model output for edge types
            X_true: (B, N) ground truth nodes
            E_true: (B, M, 3) ground truth edges [u, v, t]
            X_t: (B, N) noisy nodes at timestep t
            E_t: (B, M, 3) noisy edges at timestep t
            t: (B,) timesteps
            pt: (B,) timestep probabilities
            mask_padding: if True, don't compute loss on PAD tokens
            num_real_nodes: (B,) number of real nodes (for masking padding)
            E_mask: (B, M) mask for valid edges (for masking padding)
        Returns:
            total_loss: (B,) total loss per sample
            loss_dict: dictionary with individual losses for logging
        """
        B = X_true.shape[0]
        device = X_true.device

        # Convert logits to log probabilities
        # NOTE: Model outputs logits in format (B, spatial_dim, num_classes)
        # but we need (B, num_classes, spatial_dim) for discrete diffusion
        # Transpose to get correct format
        logits_X = logits_X.transpose(1, 2)  # (B, N, C) -> (B, C, N)
        logits_Eu = logits_Eu.transpose(1, 2)  # (B, M, C) -> (B, C, M)
        logits_Ev = logits_Ev.transpose(1, 2)  # (B, M, C) -> (B, C, M)
        logits_Et = logits_Et.transpose(1, 2)  # (B, M, C) -> (B, C, M)

        # Handle potential mismatch: Model might output max_nodes classes for u/v,
        # but we need max_nodes+1 to include MASK token at index max_nodes
        expected_uv_classes = self.schedules.max_nodes + 1
        if logits_Eu.shape[1] < expected_uv_classes:
            # Pad with one extra class (for MASK token)
            # Use mean of existing logits to avoid numerical issues
            pad_Eu = logits_Eu.mean(dim=1, keepdim=True)
            pad_Ev = logits_Ev.mean(dim=1, keepdim=True)
            logits_Eu = torch.cat([logits_Eu, pad_Eu], dim=1)
            logits_Ev = torch.cat([logits_Ev, pad_Ev], dim=1)

        log_pred_X = F.log_softmax(logits_X, dim=1)
        log_pred_Eu = F.log_softmax(logits_Eu, dim=1)
        log_pred_Ev = F.log_softmax(logits_Ev, dim=1)
        log_pred_Et = F.log_softmax(logits_Et, dim=1)

        # CRITICAL: Remap u/v logits and targets to canonical ordering (MASK at index 0)
        # Model outputs: class 0-37 = nodes, class 38 = MASK
        # We need: class 0 = MASK, class 1-38 = nodes
        #
        # Remap logits: move class 38 to position 0, shift 0-37 to positions 1-38
        if log_pred_Eu.shape[1] == self.schedules.max_nodes + 1:
            # Remap Eu logits
            log_pred_Eu_canonical = torch.cat([
                log_pred_Eu[:, -1:, :],  # MASK at position 0
                log_pred_Eu[:, :-1, :]   # Nodes at positions 1-max_nodes
            ], dim=1)
            # Remap Ev logits
            log_pred_Ev_canonical = torch.cat([
                log_pred_Ev[:, -1:, :],  # MASK at position 0
                log_pred_Ev[:, :-1, :]   # Nodes at positions 1-max_nodes
            ], dim=1)
        else:
            log_pred_Eu_canonical = log_pred_Eu
            log_pred_Ev_canonical = log_pred_Ev

        # Remap u/v targets to canonical ordering
        Eu_true_canonical = torch.where(E_true[:, :, 0] == self.mask_token_uv, 0, E_true[:, :, 0] + 1)
        Ev_true_canonical = torch.where(E_true[:, :, 1] == self.mask_token_uv, 0, E_true[:, :, 1] + 1)
        Eu_t_canonical = torch.where(E_t[:, :, 0] == self.mask_token_uv, 0, E_t[:, :, 0] + 1)
        Ev_t_canonical = torch.where(E_t[:, :, 1] == self.mask_token_uv, 0, E_t[:, :, 1] + 1)

        # Compute loss for nodes
        loss_X = self._compute_modality_loss(
            log_pred_X, X_true, X_t, t, pt,
            self.schedules.schedule_X, self.mask_token_X
        )

        # Compute loss for edges (using canonical form where MASK=0)
        if self.schedules.atomic_edges:
            # ATOMIC MODE: Loss computed on component values
            loss_Eu = self._compute_modality_loss(
                log_pred_Eu_canonical, Eu_true_canonical, Eu_t_canonical, t, pt,
                self.schedules.schedule_Eu_vals, mask_token=0  # MASK is now at 0
            )

            loss_Ev = self._compute_modality_loss(
                log_pred_Ev_canonical, Ev_true_canonical, Ev_t_canonical, t, pt,
                self.schedules.schedule_Ev_vals, mask_token=0  # MASK is now at 0
            )

            loss_Et = self._compute_modality_loss(
                log_pred_Et, E_true[:, :, 2], E_t[:, :, 2], t, pt,
                self.schedules.schedule_Et_vals, self.mask_token_t
            )
        else:
            # INDEPENDENT MODE: Loss computed independently
            loss_Eu = self._compute_modality_loss(
                log_pred_Eu_canonical, Eu_true_canonical, Eu_t_canonical, t, pt,
                self.schedules.schedule_Eu, mask_token=0  # MASK is now at 0
            )

            loss_Ev = self._compute_modality_loss(
                log_pred_Ev_canonical, Ev_true_canonical, Ev_t_canonical, t, pt,
                self.schedules.schedule_Ev, mask_token=0  # MASK is now at 0
            )

            loss_Et = self._compute_modality_loss(
                log_pred_Et, E_true[:, :, 2], E_t[:, :, 2], t, pt,
                self.schedules.schedule_Et, self.mask_token_t
            )

        # Combine losses
        total_loss = loss_X + loss_Eu + loss_Ev + loss_Et

        # For logging
        loss_dict = {
            'loss_X': loss_X.mean().item(),
            'loss_Eu': loss_Eu.mean().item(),
            'loss_Ev': loss_Ev.mean().item(),
            'loss_Et': loss_Et.mean().item(),
            'loss_total': total_loss.mean().item(),
        }

        return total_loss, loss_dict


def sample_posterior_sparse_graph(
    logits_X: torch.Tensor,
    logits_Eu: torch.Tensor,
    logits_Ev: torch.Tensor,
    logits_Et: torch.Tensor,
    X_t: torch.Tensor,
    E_t: torch.Tensor,
    t: torch.Tensor,
    schedules: SparseGraphDiffusionSchedule,
    temperature: float = 1.0,
    mask_token_uv: Optional[int] = None,
    mask_token_t: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample x_{t-1} from p(x_{t-1} | x_t) via marginalization.

    Used during generation/sampling.

    Args:
        logits_X, logits_Eu, logits_Ev, logits_Et: model outputs
        X_t, E_t: current noisy state
        t: current timestep
        schedules: multi-modal schedules
        temperature: sampling temperature
        mask_token_uv: MASK token for u/v
        mask_token_t: MASK token for edge types
    Returns:
        X_tm1: (B, N) sampled nodes at t-1
        E_tm1: (B, M, 3) sampled edges at t-1
    """
    B, N = X_t.shape
    M = E_t.shape[1]
    device = X_t.device

    if mask_token_uv is None:
        mask_token_uv = schedules.max_nodes

    # Transpose logits to get correct format (B, num_classes, spatial_dim)
    # Model outputs (B, spatial_dim, num_classes)
    logits_X = logits_X.transpose(1, 2)  # (B, N, C) -> (B, C, N)
    logits_Eu = logits_Eu.transpose(1, 2)  # (B, M, C) -> (B, C, M)
    logits_Ev = logits_Ev.transpose(1, 2)  # (B, M, C) -> (B, C, M)
    logits_Et = logits_Et.transpose(1, 2)  # (B, M, C) -> (B, C, M)

    # Handle potential mismatch: Model might output max_nodes classes for u/v,
    # but we need max_nodes+1 to include MASK token at index max_nodes
    expected_uv_classes = schedules.max_nodes + 1
    if logits_Eu.shape[1] < expected_uv_classes:
        # Pad with one extra class (for MASK token)
        # Use mean of existing logits to avoid numerical issues
        pad_Eu = logits_Eu.mean(dim=1, keepdim=True)
        pad_Ev = logits_Ev.mean(dim=1, keepdim=True)
        logits_Eu = torch.cat([logits_Eu, pad_Eu], dim=1)
        logits_Ev = torch.cat([logits_Ev, pad_Ev], dim=1)

    # Convert logits to log probs and apply temperature
    log_pred_X = F.log_softmax(logits_X / temperature, dim=1)
    log_pred_Eu = F.log_softmax(logits_Eu / temperature, dim=1)
    log_pred_Ev = F.log_softmax(logits_Ev / temperature, dim=1)
    log_pred_Et = F.log_softmax(logits_Et / temperature, dim=1)

    # CRITICAL: Remap u/v logits to canonical ordering (MASK at index 0)
    # Model outputs: class 0-37 = nodes, class 38 = MASK
    # We need: class 0 = MASK, class 1-38 = nodes
    if log_pred_Eu.shape[1] == schedules.max_nodes + 1:
        log_pred_Eu_canonical = torch.cat([
            log_pred_Eu[:, -1:, :],  # MASK at position 0
            log_pred_Eu[:, :-1, :]   # Nodes at positions 1-max_nodes
        ], dim=1)
        log_pred_Ev_canonical = torch.cat([
            log_pred_Ev[:, -1:, :],  # MASK at position 0
            log_pred_Ev[:, :-1, :]   # Nodes at positions 1-max_nodes
        ], dim=1)
    else:
        log_pred_Eu_canonical = log_pred_Eu
        log_pred_Ev_canonical = log_pred_Ev

    # Remap E_t to canonical ordering
    Eu_t_canonical = torch.where(E_t[:, :, 0] == mask_token_uv, 0, E_t[:, :, 0] + 1)
    Ev_t_canonical = torch.where(E_t[:, :, 1] == mask_token_uv, 0, E_t[:, :, 1] + 1)

    # Compute posteriors via marginalization for nodes
    log_X_t = index_to_log_onehot(X_t, schedules.num_node_types)
    log_X_tm1 = compute_posterior_with_marginalization(
        log_pred_X, log_X_t, t, schedules.schedule_X, mask_token_id=0
    )

    X_tm1 = torch.multinomial(
        log_X_tm1.exp().transpose(1, 2).reshape(-1, schedules.num_node_types), 1
    ).reshape(B, N)

    # Compute posteriors for edges (using canonical form where MASK=0)
    if schedules.atomic_edges:
        # ATOMIC MODE: Sample components, then apply atomic masking
        log_Eu_t = index_to_log_onehot(Eu_t_canonical, schedules.max_nodes + 1)
        log_Eu_tm1 = compute_posterior_with_marginalization(
            log_pred_Eu_canonical, log_Eu_t, t, schedules.schedule_Eu_vals, mask_token_id=0
        )

        log_Ev_t = index_to_log_onehot(Ev_t_canonical, schedules.max_nodes + 1)
        log_Ev_tm1 = compute_posterior_with_marginalization(
            log_pred_Ev_canonical, log_Ev_t, t, schedules.schedule_Ev_vals, mask_token_id=0
        )

        log_Et_t = index_to_log_onehot(E_t[:, :, 2], schedules.num_edge_types)
        log_Et_tm1 = compute_posterior_with_marginalization(
            log_pred_Et, log_Et_t, t, schedules.schedule_Et_vals, mask_token_id=mask_token_t
        )

        Eu_tm1 = torch.multinomial(
            log_Eu_tm1.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1
        ).reshape(B, M)

        Ev_tm1 = torch.multinomial(
            log_Ev_tm1.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1
        ).reshape(B, M)

        Et_tm1 = torch.multinomial(
            log_Et_tm1.exp().transpose(1, 2).reshape(-1, schedules.num_edge_types), 1
        ).reshape(B, M)

        # Apply atomic masking: if any component was MASK in E_t, keep all masked
        is_masked = (E_t[:, :, 2] == mask_token_t)  # Use edge type as indicator
        Eu_tm1 = torch.where(is_masked, torch.zeros_like(Eu_tm1), Eu_tm1)
        Ev_tm1 = torch.where(is_masked, torch.zeros_like(Ev_tm1), Ev_tm1)
        Et_tm1 = torch.where(is_masked, mask_token_t, Et_tm1)

    else:
        # INDEPENDENT MODE: Sample each component independently
        log_Eu_t = index_to_log_onehot(Eu_t_canonical, schedules.max_nodes + 1)
        log_Eu_tm1 = compute_posterior_with_marginalization(
            log_pred_Eu_canonical, log_Eu_t, t, schedules.schedule_Eu, mask_token_id=0
        )

        log_Ev_t = index_to_log_onehot(Ev_t_canonical, schedules.max_nodes + 1)
        log_Ev_tm1 = compute_posterior_with_marginalization(
            log_pred_Ev_canonical, log_Ev_t, t, schedules.schedule_Ev, mask_token_id=0
        )

        log_Et_t = index_to_log_onehot(E_t[:, :, 2], schedules.num_edge_types)
        log_Et_tm1 = compute_posterior_with_marginalization(
            log_pred_Et, log_Et_t, t, schedules.schedule_Et, mask_token_id=mask_token_t
        )

        Eu_tm1 = torch.multinomial(
            log_Eu_tm1.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1
        ).reshape(B, M)

        Ev_tm1 = torch.multinomial(
            log_Ev_tm1.exp().transpose(1, 2).reshape(-1, schedules.max_nodes + 1), 1
        ).reshape(B, M)

        Et_tm1 = torch.multinomial(
            log_Et_tm1.exp().transpose(1, 2).reshape(-1, schedules.num_edge_types), 1
        ).reshape(B, M)

    # CRITICAL: Enforce u < v constraint for undirected graphs
    # Since Eu and Ev are sampled independently, we need to ensure canonical form
    # Swap u and v when u >= v (but skip MASK tokens where value is 0)
    # This is in canonical coordinates where 0=MASK, 1-max_nodes=real nodes
    non_mask = (Eu_tm1 != 0) & (Ev_tm1 != 0)  # Both must be non-MASK
    needs_swap = non_mask & (Eu_tm1 >= Ev_tm1)

    # Swap u and v for edges that violate u < v
    Eu_swap = Eu_tm1.clone()
    Ev_swap = Ev_tm1.clone()
    Eu_tm1 = torch.where(needs_swap, Ev_swap, Eu_tm1)
    Ev_tm1 = torch.where(needs_swap, Eu_swap, Ev_tm1)

    # CRITICAL: Reverse remap back to original coordinate system
    # Output: {0=MASK, 1, 2, ..., max_nodes} -> {0, 1, ..., max_nodes-1, max_nodes=MASK}
    Eu_tm1 = torch.where(Eu_tm1 == 0, mask_token_uv, Eu_tm1 - 1)
    Ev_tm1 = torch.where(Ev_tm1 == 0, mask_token_uv, Ev_tm1 - 1)

    E_tm1 = torch.stack([Eu_tm1, Ev_tm1, Et_tm1], dim=-1)

    return X_tm1, E_tm1
