"""
Graph generation (report sec. 4.1.6) and conversion to NetworkX.

Three reverse processes, each kept exactly as in the original code:
- `generate_graphs(..., sampler="heuristic")`: predict x0 with temperature sampling, then re-mask
  each slot with probability 1 - alpha_{t-1} (CE runs; Table 15, Fig 9).
- `generate_graphs(..., sampler="posterior")`: report Alg. 3, sample x_{t-1} from the
  marginalized posterior (VLB config).
- `sample_edge_slots(..., sampler="heuristic_argmax")`: argmax x0 prediction, unmask masked
  slots with probability (alpha_{t-1} - alpha_t) / (1 - alpha_t). Used for Fig 8 in the
  original code, whatever the loss (temperature is ignored).
"""

import numpy as np
import networkx as nx
import torch

from ..diffusion import SparseGraphDiffusionSchedule, sample_posterior_sparse_graph


def make_alphas(diffusion_cfg, device):
    """Linear keep-probabilities used by CE masking and the heuristic samplers."""
    return torch.linspace(
        diffusion_cfg.ce_alpha_start, diffusion_cfg.ce_alpha_end, diffusion_cfg.T
    ).to(device)


def make_schedules(diffusion_cfg, dataset, device):
    """Report Eq. 29 schedules for nodes and edge slots."""
    return SparseGraphDiffusionSchedule(
        num_timesteps=diffusion_cfg.T,
        num_node_types=len(dataset.types),
        num_edge_types=len(dataset.bonds),
        max_nodes=dataset.max_length,
        atomic_edges=diffusion_cfg.atomic_edges,
        att_1=diffusion_cfg.att_1,
        att_T=diffusion_cfg.att_T,
        ctt_1=diffusion_cfg.ctt_1,
        ctt_T=diffusion_cfg.ctt_T,
    ).to(device)


def _sample(logits, temperature=1.0):
    if temperature < 1e-4:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs.view(-1, logits.shape[-1]), 1).view(logits.shape[:-1])


def _init_noise(dataset, batch, max_nodes, M, device):
    x_t = torch.full((batch, max_nodes), dataset.types.get("MASK", 0), dtype=torch.long, device=device)
    e_t = torch.zeros((batch, M, 3), dtype=torch.long, device=device)
    e_t[:, :, 0] = dataset.max_length
    e_t[:, :, 1] = dataset.max_length
    e_t[:, :, 2] = dataset.bonds.get("MASK", 0)
    num_real = torch.full((batch,), max_nodes, dtype=torch.long, device=device)
    return x_t, e_t, num_real


# ---------------------------------------------------------------------------
# NetworkX helpers
# ---------------------------------------------------------------------------

def sparse_to_nx(d_mat, valid_nodes):
    """Dense adjacency (types) -> NetworkX graph, keeping the largest connected component."""
    G = nx.Graph()
    G.add_nodes_from(range(valid_nodes))
    rows, cols = np.where(np.triu(d_mat[:valid_nodes, :valid_nodes], k=1) > 1)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c))
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def dataset_to_nx(dataset, max_samples=1000, rng=None):
    """A random subset (up to max_samples) of a dataset as NetworkX graphs."""
    rng = rng if rng is not None else np.random.default_rng(0)
    indices = rng.choice(len(dataset), size=min(len(dataset), max_samples), replace=False)
    graphs = []
    for idx in indices:
        item = dataset[int(idx)]
        A, E = item[0], item[2]
        if isinstance(E, torch.Tensor):
            E = E.numpy()
        n_real = len(A)
        G = nx.Graph()
        G.add_nodes_from(range(n_real))
        rows, cols = np.where(np.triu(E, k=1) > 1)
        for r, c in zip(rows, cols):
            if r < n_real and c < n_real:
                G.add_edge(int(r), int(c))
        graphs.append(G)
    return graphs


# ---------------------------------------------------------------------------
# Table 15 / Fig 9 sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_graphs(model, dataset, alphas, device, num_samples, batch_size, temperature=0.5,
                    sampler="heuristic", edge_slots=None, schedules=None,
                    store_raw_tensors=False):
    """
    Run the reverse process and return (graphs, raw_edge_lists, raw_tensors).

    edge_slots: number of edge slots M at generation. The original code hardcoded 100
    (config `eval.gen_edge_slots`); None means the caller passes the dataset M_max.
    raw_tensors holds (x_np, e_np) pairs when store_raw_tensors (needed for Table 14).
    """
    model.eval()
    mask_X = dataset.types.get("MASK", 0)
    mask_E = dataset.bonds.get("MASK", 0)
    uv_mask = dataset.max_length
    max_nodes = dataset.max_length
    M = int(edge_slots)
    T = len(alphas)
    if sampler == "posterior" and schedules is None:
        raise ValueError("posterior sampler needs diffusion schedules")

    generated_nx, raw_edge_lists, raw_tensors = [], [], []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in range(num_batches):
        curr_batch = min(batch_size, num_samples - len(generated_nx))
        if curr_batch <= 0:
            break
        x_t, e_t, num_real = _init_noise(dataset, curr_batch, max_nodes, M, device)

        if sampler == "posterior":
            for t in reversed(range(T)):
                t_tens = torch.full((curr_batch,), t, device=device, dtype=torch.long)
                logits_X, logits_Eu, logits_Ev, logits_Et = model(x_t, e_t, t_tens, num_real)
                if t > 0:
                    x_t, e_t = sample_posterior_sparse_graph(
                        logits_X, logits_Eu, logits_Ev, logits_Et,
                        x_t, e_t, t_tens, schedules,
                        temperature=temperature, mask_token_uv=uv_mask, mask_token_t=mask_E,
                    )
                else:
                    x_t = _sample(logits_X, temperature)
                    eu = _sample(logits_Eu, temperature)
                    ev = _sample(logits_Ev, temperature)
                    et = _sample(logits_Et, temperature)
                    e_t = torch.stack([eu, ev, et], dim=-1)
        else:
            for t in reversed(range(T)):
                t_tens = torch.full((curr_batch,), t, device=device, dtype=torch.long)
                logits_X, logits_Eu, logits_Ev, logits_Et = model(x_t, e_t, t_tens, num_real)
                x0_pred = _sample(logits_X, temperature)
                eu_pred = _sample(logits_Eu, temperature)
                ev_pred = _sample(logits_Ev, temperature)
                et_pred = _sample(logits_Et, temperature)
                if t > 0:
                    alpha_next = alphas[t - 1]
                    remask_X = torch.rand_like(x_t, dtype=torch.float) > alpha_next
                    x_next = x0_pred.clone()
                    x_next[remask_X] = mask_X
                    remask_E = torch.rand(curr_batch, M, device=device) > alpha_next
                    e_next = torch.stack([eu_pred, ev_pred, et_pred], dim=-1)
                    e_flat = e_next.view(-1, 3)
                    m_flat = remask_E.view(-1)
                    e_flat[m_flat, 0] = uv_mask
                    e_flat[m_flat, 1] = uv_mask
                    e_flat[m_flat, 2] = mask_E
                    e_t = e_flat.view(curr_batch, M, 3)
                    x_t = x_next
                else:
                    x_t = x0_pred
                    e_t = torch.stack([eu_pred, ev_pred, et_pred], dim=-1)

        # Post-processing: mask nodes without edges, build dense adjacency, convert to NX
        for i in range(curr_batch):
            e_i = e_t[i]
            valid_edges_mask = e_i[:, 2] > 1
            nodes_with_edges = set()
            if valid_edges_mask.any():
                for idx in torch.where(valid_edges_mask)[0]:
                    u = e_i[idx, 0].item()
                    v = e_i[idx, 1].item()
                    if u < max_nodes and u != uv_mask:
                        nodes_with_edges.add(u)
                    if v < max_nodes and v != uv_mask:
                        nodes_with_edges.add(v)
                for node_idx in range(max_nodes):
                    if node_idx not in nodes_with_edges and x_t[i, node_idx].item() != mask_X:
                        x_t[i, node_idx] = mask_X

            x_np = x_t[i].cpu().numpy()
            e_np = e_i.cpu().numpy()
            d_mat = np.zeros((max_nodes, max_nodes), dtype=int)
            for u, v, tp in e_np:
                if u < max_nodes and v < max_nodes and tp > 1:
                    d_mat[int(u), int(v)] = int(tp)
                    d_mat[int(v), int(u)] = int(tp)

            valid_nodes = 0
            for node_idx in range(max_nodes):
                nt = x_np[node_idx]
                if (nt != mask_X and nt != dataset.types.get("PAD", -1)
                        and node_idx in nodes_with_edges):
                    valid_nodes = node_idx + 1

            raw_edges = [
                (int(u), int(v)) for u, v, tp in e_np
                if tp > 1 and u < max_nodes and v < max_nodes and u != uv_mask and v != uv_mask
            ]
            raw_edge_lists.append(raw_edges)
            if store_raw_tensors:
                raw_tensors.append((x_np.copy(), e_np.copy()))
            generated_nx.append(sparse_to_nx(d_mat, valid_nodes))

    return generated_nx, raw_edge_lists, raw_tensors


# ---------------------------------------------------------------------------
# Fig 8 sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_edge_slots(model, dataset, alphas, device, num_samples, M_max,
                      sampler="heuristic_argmax", temperature=0.5, schedules=None,
                      batch_size=32):
    """Return a list of generated edge-slot tensors (B, M_max, 3), one per batch (Fig 8)."""
    model.eval()
    N_max = dataset.max_length
    mask_X = dataset.types.get("MASK", 0)
    mask_E = dataset.bonds.get("MASK", 0)
    uv_mask = dataset.max_length
    T = len(alphas)

    batches = []
    done = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    for _ in range(num_batches):
        current_bs = min(batch_size, num_samples - done)
        if current_bs <= 0:
            break
        x_t, e_t, num_real = _init_noise(dataset, current_bs, N_max, M_max, device)

        if sampler == "posterior":
            for t in reversed(range(T)):
                t_tens = torch.full((current_bs,), t, device=device, dtype=torch.long)
                logits_X, logits_Eu, logits_Ev, logits_Et = model(x_t, e_t, t_tens, num_real)
                if t > 0:
                    x_t, e_t = sample_posterior_sparse_graph(
                        logits_X, logits_Eu, logits_Ev, logits_Et,
                        x_t, e_t, t_tens, schedules,
                        temperature=temperature, mask_token_uv=uv_mask, mask_token_t=mask_E,
                    )
                else:
                    x_t = _sample(logits_X, temperature)
                    eu = _sample(logits_Eu, temperature)
                    ev = _sample(logits_Ev, temperature)
                    et = _sample(logits_Et, temperature)
                    # Enforce u < v (skip MASK pointers)
                    non_mask = (eu != uv_mask) & (ev != uv_mask)
                    swap = non_mask & (eu >= ev)
                    eu, ev = torch.where(swap, ev, eu), torch.where(swap, eu, ev)
                    e_t = torch.stack([eu, ev, et], dim=-1)
            E_curr = e_t
        else:
            timesteps = list(range(T))[::-1]
            X_curr, E_curr = x_t, e_t
            for i, t_val in enumerate(timesteps):
                t_tensor = torch.full((current_bs,), t_val, device=device, dtype=torch.long)
                logits_X, logits_Eu, logits_Ev, logits_Et = model(X_curr, E_curr, t_tensor, num_real)
                X_pred = logits_X.argmax(dim=-1)
                E_pred = torch.stack([
                    logits_Eu.argmax(dim=-1),
                    logits_Ev.argmax(dim=-1),
                    logits_Et.argmax(dim=-1),
                ], dim=-1)

                if i < len(timesteps) - 1:
                    alpha_curr = alphas[t_val]
                    alpha_next = alphas[timesteps[i + 1]]
                    p_unmask = ((alpha_next - alpha_curr) / (1.0 - alpha_curr + 1e-8)).clamp(0, 1)

                    masked_x = X_curr == mask_X
                    unmask_x = (torch.rand_like(X_curr, dtype=torch.float) < p_unmask) & masked_x
                    X_curr[unmask_x] = X_pred[unmask_x]

                    masked_e = E_curr[:, :, 2] == mask_E
                    unmask_e = (torch.rand((current_bs, M_max), device=device) < p_unmask) & masked_e
                    E_curr[unmask_e] = E_pred[unmask_e]

            masked_e = E_curr[:, :, 2] == mask_E
            E_curr[masked_e] = E_pred[masked_e]

        batches.append(E_curr)
        done += current_bs
    return batches


def unique_edge_counts(edge_batches, N_max):
    """Number of unique, non-self-loop edges (type > NO_EDGE, u, v < N_max) per generated graph."""
    counts = []
    for E_b in edge_batches:
        for e_graph in E_b:
            valid = (e_graph[:, 2] > 1) & (e_graph[:, 0] < N_max) & (e_graph[:, 1] < N_max)
            edges = set()
            for idx in torch.nonzero(valid).squeeze(-1):
                u = e_graph[idx, 0].item()
                v = e_graph[idx, 1].item()
                if u > v:
                    u, v = v, u
                if u != v:
                    edges.add((u, v))
            counts.append(len(edges))
    return counts
