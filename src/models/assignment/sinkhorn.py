"""Sparse Sinkhorn projection onto the polystochastic assignment polytope.

Avoids materializing the dense ``(Bp, m_ot^V)`` tensor by working directly with
the flat edge list and computing per-view marginals via scatter operations.
Per-iteration cost: ``O(M · V)`` time, ``O(M + Bp·V·N)`` memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_DUST_EPS_MASS = 1.0e-3  # mass reserved for the per-view dustbin


def _scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Numerically-stable logsumexp grouped by ``index``, using scatter ops."""
    # Max per group, detached — the shift is constant for gradient purposes and
    # detaching avoids relying on scatter_reduce's amax autograd.
    with torch.no_grad():
        max_val = torch.full((dim_size,), float("-inf"), device=src.device, dtype=src.dtype)
        max_val.scatter_reduce_(0, index, src, reduce="amax", include_self=True)

    max_gathered = max_val[index]
    ok = torch.isfinite(max_gathered)
    safe_max = torch.where(ok, max_gathered, torch.zeros_like(max_gathered))
    exp_shifted = torch.exp(src - safe_max)

    sum_exp = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    sum_exp.scatter_add_(0, index, exp_shifted)

    # Clamp before log to avoid log(0) = -inf, whose gradient (1/0 = inf) yields
    # NaN through 0 * inf in autograd even when the upstream gradient is zero.
    # For non-empty groups sum_exp >= 1 (contains exp(0)), so the clamp is inert.
    return torch.log(sum_exp.clamp(min=1e-30)) + max_val


def _compute_marginal_targets(
    node_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capacity-mode Sinkhorn marginal targets.

    Each view gets a per-node marginal of ``1/N`` and a dustbin slot that
    absorbs the leftover ``(N − n_active)/N + dust_eps``. Batches with fewer
    than two non-empty views collapse to an all-dustbin solution.

    Returns:
        log_mu: ``(Bp, V, m_ot)`` target log-marginals.
        active: ``(Bp, V, m_ot)`` bool — finite marginal entries.
        valid_ext: ``(Bp, V, m_ot)`` bool — valid node mask incl. dustbin.
    """
    Bp, V, N = node_mask.shape
    m_ot = N + 1
    dustbin_idx = N
    K64 = torch.tensor(float(N), device=device, dtype=torch.float64)
    dust_eps = torch.tensor(_DUST_EPS_MASS, device=device, dtype=torch.float64)

    n_v = node_mask.sum(dim=2).to(torch.float64)  # (Bp, V)
    nonempty_views = (n_v > 0).sum(dim=1)          # (Bp,)

    dust_raw = (K64 - n_v).clamp(min=0.0) + dust_eps  # (Bp, V)
    norm = K64.expand_as(n_v)
    dust_allowed = dust_raw > 0.0

    log_mu = torch.full((Bp, V, m_ot), float("-inf"), device=device, dtype=torch.float64)
    log_real = -torch.log(norm.clamp_min(1e-12))
    log_mu[:, :, :N] = torch.where(
        node_mask,
        log_real.unsqueeze(-1),
        torch.full((Bp, V, N), float("-inf"), device=device, dtype=torch.float64),
    )
    log_mu[:, :, dustbin_idx] = torch.where(
        dust_allowed,
        torch.log(dust_raw.clamp_min(1e-12)) - torch.log(norm.clamp_min(1e-12)),
        torch.full((Bp, V), float("-inf"), device=device, dtype=torch.float64),
    )

    bad = nonempty_views < 2
    if bad.any():
        log_mu[bad, :, :N] = float("-inf")
        log_mu[bad, :, dustbin_idx] = 0.0

    active = torch.isfinite(log_mu)
    valid_ext = torch.ones((Bp, V, m_ot), device=device, dtype=torch.bool)
    valid_ext[:, :, :N] = node_mask
    valid_ext[:, :, dustbin_idx] = torch.where(bad.view(Bp, 1), torch.ones_like(dust_allowed), dust_allowed)

    return log_mu, active, valid_ext


class SparseSinkhornSolver(nn.Module):
    """Sparse Sinkhorn projection onto the polystochastic polytope.

    Each call projects ``log_E`` (unnormalized log-probabilities, one per
    hyperedge) onto the feasible set where the per-view marginals match the
    capacity targets from :func:`_compute_marginal_targets`. The dustbin prior
    ``alpha`` is added to each missing view's contribution before iteration.
    """

    def __init__(self, num_views: int):
        super().__init__()
        self.num_views = int(num_views)

    def forward(
        self,
        log_E: torch.Tensor,          # (M_edges,)
        edge_tuples: torch.Tensor,    # (M_edges, 1+V) [batch, n_0, …, n_{V-1}]
        node_mask: torch.Tensor,       # (Bp, V, N)
        alpha: torch.Tensor,           # scalar dustbin prior
        iters: int,
    ) -> torch.Tensor:
        device = log_E.device
        Bp, V, N = node_mask.shape
        M_edges = int(log_E.numel())

        if M_edges == 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        m_ot = N + 1
        dustbin_idx = N

        log_mu, active, valid_ext = _compute_marginal_targets(node_mask, device)

        # Dustbin prior added per missing view of each edge.
        alpha64 = alpha.to(torch.float64).reshape(())
        is_dust = edge_tuples[:, 1:] == dustbin_idx  # (M_edges, V)
        dust_cnt = is_dust.sum(dim=1).to(torch.float64)
        log_E_64 = log_E.to(torch.float64) + (alpha64 * dust_cnt)

        # Fallback all-dustbin edges (feasibility guarantee) for batches that
        # don't already have one — suppressed otherwise.
        is_all_dust = is_dust.all(dim=1)
        batch_has_dust = torch.zeros(Bp, dtype=torch.bool, device=device)
        if is_all_dust.any():
            batch_has_dust.scatter_(0, edge_tuples[is_all_dust, 0], True)

        fb_edges = torch.full((Bp, 1 + V), dustbin_idx, device=device, dtype=edge_tuples.dtype)
        fb_edges[:, 0] = torch.arange(Bp, device=device, dtype=edge_tuples.dtype)
        fb_logits = torch.where(
            batch_has_dust,
            torch.tensor(float("-inf"), device=device, dtype=torch.float64),
            (alpha64 * float(V)).expand(Bp),
        )

        all_edges = torch.cat([edge_tuples, fb_edges], dim=0)
        logits = torch.cat([log_E_64, fb_logits], dim=0)

        # Mask invalid edges (any view's node invalid → -inf).
        edge_batch = all_edges[:, 0]
        edge_nodes = all_edges[:, 1:]
        invalid = torch.zeros(all_edges.shape[0], device=device, dtype=torch.bool)
        for v in range(V):
            invalid |= ~valid_ext[edge_batch, v, edge_nodes[:, v]]
        logits = torch.where(invalid, torch.tensor(float("-inf"), device=device, dtype=torch.float64), logits)

        # Pre-compute per-view group indices and per-view targets.
        dim_size = Bp * m_ot
        view_group_idx: list[torch.Tensor] = []
        view_target: list[torch.Tensor] = []
        view_active: list[torch.Tensor] = []
        for v in range(V):
            g = edge_batch * m_ot + edge_nodes[:, v]
            view_group_idx.append(g)
            view_target.append(log_mu[:, v].reshape(-1)[g])
            view_active.append(active[:, v].reshape(-1)[g])

        # Sparse Sinkhorn iterations.
        for _ in range(iters):
            for v in range(V):
                marginal = _scatter_logsumexp(logits, view_group_idx[v], dim_size)
                marginal_e = marginal[view_group_idx[v]]
                ok = torch.isfinite(marginal_e)
                delta = view_target[v] - marginal_e
                logits = torch.where(view_active[v] & ok, logits + delta, logits)

        probs = torch.exp(logits).to(torch.float32)
        return probs[:M_edges]
