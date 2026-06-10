import math

import torch
import torch.nn as nn


def _cosine_alpha_bar(
    T: int,
    s: float = 0.008,
    min_alpha_bar: float = 1e-4,
    device: torch.device | None = None,
) -> torch.Tensor:
    if T <= 1:
        return torch.ones((T,), dtype=torch.float32, device=device)

    x = torch.linspace(0, T - 1, T, device=device, dtype=torch.float32)
    t = x / (T - 1)
    f = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    f = f / f[0].clamp_min(1e-12)
    return f.clamp(min=min_alpha_bar, max=1.0)


def _extract_1d(arr: torch.Tensor, t: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
    return arr.gather(0, t.view(-1)).view(out_shape)


class ProjectedConstrainedDDIM(nn.Module):
    """Projected DDIM diffusion on the assignment polytope.

    Latent embedding u = log(x). Forward chain corrupts u_0 with one-shot
    Gaussian noise and projects through Sinkhorn back onto the feasible
    polytope. Reverse update is deterministic DDIM (eta = 0) on u, with a
    Sinkhorn projection at every step so the trajectory stays on the polytope.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        sinkhorn_solver: nn.Module,
        alpha_param: nn.Parameter,
        num_timesteps: int,
        sampling_timesteps: int,
        skh_iterations: int,
        u_clip: float = 30.0,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.sinkhorn_solver = sinkhorn_solver
        self.alpha = alpha_param

        self.num_timesteps = int(num_timesteps)
        self.sampling_timesteps = int(sampling_timesteps)
        self.skh_iterations = int(skh_iterations)

        self.register_buffer("log_eps", torch.tensor(1e-12, dtype=torch.float32), persistent=False)
        # Frozen zero-init Parameter kept only for state_dict backward-compatibility
        # with older checkpoints that trained a latent temperature here. The math no
        # longer reads it.
        self.log_tau = nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)
        self.u_clip = u_clip

        self._init_schedule(self.num_timesteps)

    def _init_schedule(self, T: int) -> None:
        device = next(self.parameters()).device
        ab = _cosine_alpha_bar(T, s=0.008, min_alpha_bar=1e-4, device=device).to(torch.float32)
        self.register_buffer("alpha_bar", ab, persistent=False)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(ab), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt((1.0 - ab).clamp_min(0.0)), persistent=False)

    # ---- projection / embedding ----

    def project_u_to_x(
        self,
        u: torch.Tensor,
        graph: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
        iters: int | None = None,
    ) -> torch.Tensor:
        """Project latent u to feasible probability space x via Sinkhorn."""
        return self.sinkhorn_solver(
            log_E=u,
            edge_tuples=graph["edge_tuples"],
            node_mask=node_mask,
            alpha=self.alpha,
            iters=int(self.skh_iterations if iters is None else iters),
        ).to(torch.float32)

    def embed_x_to_u(self, x: torch.Tensor) -> torch.Tensor:
        """Embed feasible x to latent space u = log(x)."""
        u = torch.log(x.clamp_min(self.log_eps))  # type: ignore[arg-type]
        return u.to(torch.float32).clamp(-self.u_clip, self.u_clip)

    def predict_u0_from_xt(
        self,
        x_t: torch.Tensor,
        t_batch: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the raw denoiser output in latent units."""
        node_X = graph["X"]
        if node_X.dim() == 3:
            if node_X.shape[1] != 1:
                raise ValueError(f"Expected a single joint slice per node, got node features {tuple(node_X.shape)}")
            node_X = node_X[:, 0]
        elif node_X.dim() != 2:
            raise ValueError(f"Unexpected node feature shape: {tuple(node_X.shape)}")
        ab_t = self.alpha_bar[t_batch.long()].to(torch.float32)
        noise_level = torch.sqrt((1.0 - ab_t).clamp_min(0.0))
        u0_hat = self.denoiser(
            node_X,
            x_t,
            graph["z_cue"],
            t_batch.to(dtype=torch.float32),
            graph["hedge_idx"],
            graph["node_batch"],
            graph["edge_batch"],
            noise_level=noise_level,
        )
        return u0_hat.clamp(-self.u_clip, self.u_clip)

    # ---- training corruption ----

    def _sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.num_timesteps <= 1:
            return torch.zeros((batch_size,), device=device, dtype=torch.long)
        return torch.randint(1, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(
        self,
        u0: torch.Tensor,
        t_edge: torch.Tensor,
        graph: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
        proj_iters: int,
    ) -> torch.Tensor:
        """One-shot Gaussian corruption: u_t = sqrt(α̅_t)·u_0 + sqrt(1-α̅_t)·ε, then project."""
        if u0.numel() == 0:
            return torch.empty_like(u0)

        eps = torch.randn_like(u0)
        sqrt_ab = _extract_1d(self.sqrt_alpha_bar, t_edge, u0.shape).to(torch.float32)
        sqrt_omab = _extract_1d(self.sqrt_one_minus_alpha_bar, t_edge, u0.shape).to(torch.float32)
        u_t_raw = sqrt_ab * u0 + sqrt_omab * eps
        return self.project_u_to_x(u_t_raw, graph, node_mask, iters=proj_iters)

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        t_batch: torch.Tensor,
        graph: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
        proj_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u0_hat_raw = self.predict_u0_from_xt(x_t=x_t, t_batch=t_batch, graph=graph)
        x0_hat = self.project_u_to_x(u0_hat_raw, graph, node_mask, iters=proj_iters)
        u0_hat_projected = self.embed_x_to_u(x0_hat)
        return x0_hat, u0_hat_projected

    # ---- sampling schedule ----

    def make_sampling_timesteps(self) -> list[int]:
        T = self.num_timesteps
        S = self.sampling_timesteps
        ts = torch.linspace(0, T - 1, S + 1)
        ts = torch.round(ts).long().clamp(0, T - 1)
        ts = torch.unique_consecutive(ts)

        if ts.numel() == 0 or ts[0].item() != 0:
            ts = torch.cat([torch.zeros(1, dtype=torch.long), ts])
        if ts[-1].item() != (T - 1):
            ts = torch.cat([ts, torch.tensor([T - 1], dtype=torch.long)])

        ts = torch.unique_consecutive(ts)
        return ts.flip(0).tolist()

    # ---- training ----

    def forward_train(
        self,
        graph: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
        E0_bin: torch.Tensor,
        Bp: int,
        e0_logit_scale: float,
    ) -> dict[str, torch.Tensor]:
        """
        One-shot Gaussian-corruption training pass.
        """
        device = node_mask.device

        if graph["M"] == 0:
            return {}

        u0_logits = torch.where(
            E0_bin > 0,
            torch.tensor(e0_logit_scale, device=device),
            torch.tensor(-e0_logit_scale, device=device),
        )
        with torch.no_grad():
            x0_gt = self.project_u_to_x(u0_logits, graph, node_mask, iters=self.skh_iterations)
            u0 = self.embed_x_to_u(x0_gt)

        t = self._sample_timestep(batch_size=Bp, device=device)
        t_edge = t[graph["edge_batch"]]
        x_t = self.q_sample(
            u0=u0,
            t_edge=t_edge,
            graph=graph,
            node_mask=node_mask,
            proj_iters=max(1, self.skh_iterations // 2),
        )
        x0_hat, u0_hat = self._predict_x0(
            x_t=x_t,
            t_batch=t,
            graph=graph,
            node_mask=node_mask,
            proj_iters=max(1, self.skh_iterations // 2),
        )
        return {
            "x0_hat": x0_hat,
            "x0_gt": x0_gt.detach(),
            "u0_hat": u0_hat,
            "u0": u0.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        graph: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
        Bp: int,
    ) -> torch.Tensor:
        """
        Projected DDIM reverse sampling.
        """
        device = node_mask.device
        M = int(graph["M"])
        if M == 0:
            return torch.empty((0,), device=device, dtype=torch.float32)

        proj_iters = int(self.skh_iterations)
        proj_iters_inner = max(1, proj_iters // 2)

        times = self.make_sampling_timesteps()

        x_t = self.project_u_to_x(
            torch.zeros((M,), device=device, dtype=torch.float32),
            graph,
            node_mask,
            iters=proj_iters_inner,
        )
        u_t = self.embed_x_to_u(x_t)

        for t_curr, t_next in zip(times[:-1], times[1:]):
            t_batch = torch.full((Bp,), int(t_curr), device=device, dtype=torch.long)
            t_edge = t_batch[graph["edge_batch"]]

            u0_hat_raw = self.predict_u0_from_xt(x_t=x_t, t_batch=t_batch, graph=graph)
            x0_hat = self.project_u_to_x(u0_hat_raw, graph, node_mask, iters=proj_iters_inner)
            u0_hat = self.embed_x_to_u(x0_hat)

            ab_t = _extract_1d(self.alpha_bar, t_edge, u_t.shape).to(torch.float32).clamp_min(1e-8)
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_omab_t = torch.sqrt((1.0 - ab_t).clamp_min(1e-8))
            eps_hat = (u_t - sqrt_ab_t * u0_hat) / sqrt_omab_t

            ab_prev_val = float(self.alpha_bar[int(t_next)])
            ab_prev = torch.full_like(ab_t, ab_prev_val).clamp_min(1e-8)
            u_prev = torch.sqrt(ab_prev) * u0_hat + torch.sqrt((1.0 - ab_prev).clamp_min(0.0)) * eps_hat

            x_t = self.project_u_to_x(u_prev, graph, node_mask, iters=proj_iters_inner)
            u_t = self.embed_x_to_u(x_t)

        return self.project_u_to_x(u_t, graph, node_mask, iters=proj_iters)
