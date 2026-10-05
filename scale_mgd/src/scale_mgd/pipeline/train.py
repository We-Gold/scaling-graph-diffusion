"""
Training loop (report sec. 4.1.5, Alg. 2) with two loss modes.

- "ce": Bernoulli masking with keep-probability alpha_t = linspace(0.9999, 0.0001, T)[t] on all
  node slots of real nodes and all M_max edge slots (u, v, tau masked together), then plain
  cross-entropy on x0 (PAD edge pointers ignored). This is what the 2026-03-03 report runs used.
- "vlb": forward process q(x_t | x_0) of report Eq. 29 with atomic edge masking, then the VLB
  with x0 marginalization plus the adaptive auxiliary loss (report Eq. 33-36).

Timesteps are drawn uniformly from [1, T-1] (the original "t=0 bug" fix is kept).
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..data import TruncationCounter, edge_type_marginals, training_collate_fn
from ..diffusion import SparseGraphDiffusionLoss, q_sample_sparse_graph
from .runner import PipelineStep


def apply_ce_masking(X_b, E_b, num_real_b, alpha_b, mask_X, mask_E_uv, mask_E_type):
    """Bernoulli masking used by the CE runs. Each slot is masked with probability 1 - alpha."""
    device = X_b.device
    X_noisy = X_b.clone()
    E_noisy = E_b.clone()
    n_slots = E_b.shape[1]
    for b in range(X_b.shape[0]):
        n_real = num_real_b[b].item()
        mask_x = torch.rand(n_real, device=device) > alpha_b[b]
        X_noisy[b, :n_real][mask_x] = mask_X
        if n_slots > 0:
            mask_e = torch.rand(n_slots, device=device) > alpha_b[b]
            E_noisy[b, :n_slots][mask_e, 0] = mask_E_uv
            E_noisy[b, :n_slots][mask_e, 1] = mask_E_uv
            E_noisy[b, :n_slots][mask_e, 2] = mask_E_type
    return X_noisy, E_noisy


class FullTrainingStep(PipelineStep):
    """Train the model in context on context['dataset'] and save the final state dict."""

    def __init__(self, cfg, batch_size, train_fraction, checkpoint_dir):
        super().__init__("Training")
        self.cfg = cfg
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.checkpoint_dir = Path(checkpoint_dir)

    def execute(self, context):
        cfg = self.cfg
        model = context["model"]
        device = context["device"]
        dataset = context["dataset"]
        T = cfg.diffusion.T
        loss_mode = cfg.diffusion.loss

        dataset_size = len(dataset)
        if self.train_fraction is not None:
            subset_size = int(dataset_size * self.train_fraction)
            subset = Subset(dataset, list(range(subset_size)))
            print(f"Training on the first {subset_size}/{dataset_size} graphs "
                  f"({self.train_fraction * 100:.1f}%)")
        else:
            subset_size = dataset_size
            subset = dataset
            print(f"Training on all {subset_size} graphs")

        N_max = dataset.max_length
        M_max = context["M_max"]
        no_edge = dataset.bonds.get("NO_BOND", 1)
        truncation = TruncationCounter()

        def collate(batch):
            return training_collate_fn(
                batch, N_max, M_max,
                node_pad=dataset.types["PAD"],
                edge_pad=(no_edge, N_max, N_max),
                truncation=truncation,
            )

        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True, collate_fn=collate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        loss_fn = None
        if loss_mode == "vlb":
            schedules = context["schedules"]
            loss_fn = SparseGraphDiffusionLoss(
                schedules=schedules,
                auxiliary_loss_weight=cfg.diffusion.aux_loss_weight,
                adaptive_auxiliary_loss=cfg.diffusion.adaptive_aux,
                mask_token_X=dataset.types.get("MASK", 0),
                mask_token_uv=N_max,
                mask_token_t=dataset.bonds.get("MASK", 0),
            )
        print(f"Loss mode: {loss_mode}, epochs {cfg.train.epochs}, batch size {self.batch_size}")

        marginal_distributions = {}
        if cfg.train.use_edge_type_marginals:
            probs_t = edge_type_marginals(dataset, M_max, subset_size)
            marginal_distributions["t"] = probs_t.to(device)
            print(f"Edge type marginals (training only): {probs_t.tolist()}")

        mask_X = dataset.types.get("MASK", 0)
        mask_E_type = dataset.bonds.get("MASK", 0)
        mask_E_uv = N_max  # virtual node index
        alphas = context["alphas"]

        losses, epoch_times = [], []
        global_step = 0
        for epoch in range(1, cfg.train.epochs + 1):
            epoch_start = time.time()
            model.train()
            epoch_loss, steps_in_epoch = 0.0, 0

            for X_b, E_b, _E_mask_b, num_real_b in dataloader:
                X_b, E_b, num_real_b = X_b.to(device), E_b.to(device), num_real_b.to(device)
                B = X_b.shape[0]
                optimizer.zero_grad()

                t_batch = torch.randint(1, T, (B,), device=device)
                if loss_mode == "vlb":
                    X_noisy, E_noisy = q_sample_sparse_graph(
                        X_b, E_b, t_batch, context["schedules"],
                        mask_token_uv=mask_E_uv, mask_token_t=mask_E_type,
                    )
                else:
                    X_noisy, E_noisy = apply_ce_masking(
                        X_b, E_b, num_real_b, alphas[t_batch], mask_X, mask_E_uv, mask_E_type
                    )

                kwargs = {"num_real_nodes": num_real_b}
                if marginal_distributions:
                    kwargs["marginal_distributions"] = marginal_distributions
                logits_X, logits_Eu, logits_Ev, logits_Et = model(X_noisy, E_noisy, t_batch, **kwargs)

                if loss_mode == "vlb":
                    pt = torch.ones(B, device=device) / T
                    loss_per_sample, _ = loss_fn.compute_loss(
                        logits_X, logits_Eu, logits_Ev, logits_Et,
                        X_b, E_b, X_noisy, E_noisy, t_batch, pt,
                    )
                    loss = loss_per_sample.mean()
                else:
                    # ignore_index=N_max drops PAD edge pointers from the u/v loss
                    loss = (
                        F.cross_entropy(logits_X.transpose(1, 2), X_b)
                        + F.cross_entropy(logits_Eu.transpose(1, 2), E_b[..., 0], ignore_index=N_max)
                        + F.cross_entropy(logits_Ev.transpose(1, 2), E_b[..., 1], ignore_index=N_max)
                        + F.cross_entropy(logits_Et.transpose(1, 2), E_b[..., 2])
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

                loss_val = loss.item()
                losses.append(loss_val)
                epoch_loss += loss_val
                steps_in_epoch += 1
                global_step += 1
                if global_step % cfg.train.log_interval == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss {loss_val:.4f}")

            epoch_times.append(time.time() - epoch_start)
            print(f"End epoch {epoch} | avg loss {epoch_loss / max(1, steps_in_epoch):.4f} "
                  f"| {epoch_times[-1]:.1f}s")

        if truncation.graphs_cut:
            print(f"WARNING: edges were cut at M_max = {M_max} in {truncation.graphs_cut} graph "
                  f"samples (summed over {cfg.train.epochs} epochs).")

        final_path = self.checkpoint_dir / "model_final.pt"
        torch.save(model.state_dict(), final_path)
        with open(self.checkpoint_dir / "losses.json", "w") as f:
            json.dump(losses, f)
        print(f"Saved {final_path}")

        context["loss_history"] = losses
        context["train_time_per_epoch"] = epoch_times
        context["graphs_cut_at_M_max"] = truncation.graphs_cut
        if device.type == "cuda":
            context["peak_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        else:
            context["peak_memory_mb"] = None
