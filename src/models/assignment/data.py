import torch


def get_matching_data(
    instances_dict: dict[str, torch.Tensor],
    drop_prob: float = 0.2,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """
    Prepare training data for multi-view correspondence matching.

    This function:
    1. Augments data by randomly dropping one view per person (when >2 views visible)
    2. Extracts GT correspondences from person IDs

    Args:
        instances_dict: Dictionary containing:
            - score: (B, V, N, J, D) detection scores
            - person_ids: (B, V, N, J, D) person IDs (uses [..., 0, 0])
            - image: (B, V, N, J, 2) 2D positions
            - other tensors with same (B, V, N, ...) prefix
        drop_prob: Probability of dropping one view per person for augmentation

    Returns:
        dict containing:
            - instances: Augmented detection tensors (same structure as input, minus person_ids)
            - valid_mask: (B, V, N) boolean mask of valid detections
            - gt_correspondences: (K, 1+V) tensor where each row is
              [batch_idx, n_0, n_1, ..., n_{V-1}], with N as dustbin for missing views
    """
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")

    # Deep copy tensors so augmentation doesn't mutate the caller's data
    instances = {k: v.clone() for k, v in instances_dict.items()}
    pid = instances["person_ids"][..., 0]  # (B, V, N)
    valid = instances["score"][..., 0, 0] > 0  # (B, V, N)

    # --- Augmentation: randomly drop one view per person ---
    if drop_prob > 0.0:
        pid, valid = _random_view_dropout(instances, pid, valid, drop_prob)

    # --- Build GT correspondences ---
    gt_correspondences = _build_gt_correspondences(pid, valid)

    return {
        "instances": instances,
        "valid_mask": valid,
        "gt_correspondences": gt_correspondences,
    }


def _random_view_dropout(
    instances: dict[str, torch.Tensor],
    pid: torch.Tensor,  # (B, V, N)
    valid: torch.Tensor,  # (B, V, N)
    drop_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each person visible in >2 views, randomly drop one view with probability drop_prob.
    """
    B, V, N = pid.shape
    device = pid.device

    unique_pids = pid.unique()
    unique_pids = unique_pids[unique_pids >= 0]
    P = len(unique_pids)

    if P == 0:
        return pid, valid

    # Build (P, B, V, N) mask for each person
    pid_masks = (pid.unsqueeze(0) == unique_pids.view(P, 1, 1, 1)) & valid.unsqueeze(0)
    views_per_person = pid_masks.any(dim=3).sum(dim=2)  # (P, B)

    # Eligible: >2 views and passes probability check
    eligible = (views_per_person > 2) & (torch.rand(P, B, device=device) < drop_prob)

    if not eligible.any():
        return pid, valid

    # Random view selection: noise + mask invalid + argmax
    view_presence = pid_masks.any(dim=3)  # (P, B, V)
    rand_scores = torch.rand(P, B, V, device=device)
    rand_scores[~view_presence] = -1
    chosen_views = rand_scores.argmax(dim=2)  # (P, B)

    # Find detection index for chosen view
    v_idx = chosen_views.view(P, B, 1, 1).expand(-1, -1, 1, N)
    chosen_n = torch.gather(pid_masks, 2, v_idx).squeeze(2).int().argmax(dim=2)  # (P, B)

    # Apply dropout using advanced indexing
    elig_coords = eligible.nonzero(as_tuple=False)
    if elig_coords.numel() > 0:
        p_idx, b_idx = elig_coords[:, 0], elig_coords[:, 1]
        v_idx, n_idx = chosen_views[p_idx, b_idx], chosen_n[p_idx, b_idx]

        for t in instances.values():
            t[b_idx, v_idx, n_idx] = 0
        pid[b_idx, v_idx, n_idx] = -1
        valid[b_idx, v_idx, n_idx] = False

    return pid, valid


def _build_gt_correspondences(
    pid: torch.Tensor,  # (B, V, N)
    valid: torch.Tensor,  # (B, V, N)
) -> torch.Tensor:
    """
    Build GT correspondence matrix from person IDs.

    - For each valid pid >= 0: one row that matches that person across views.
    - Valid detections with pid < 0 are left unlabeled (no GT row) and act only
      as negative candidates.

    Returns: (K, V + 1) tensor where each row is [batch_idx, n_0, ..., n_{V-1}]
             Detection indices are 0..N-1, or N (dustbin) for missing views.
    """
    B, V, N = pid.shape
    device = pid.device
    gt_rows = []

    # --- (A) Standard multi-view GT for labeled persons (pid >= 0) ---
    for b in range(B):
        valid_pids = pid[b][valid[b]].unique()
        valid_pids = valid_pids[valid_pids >= 0]

        for p in valid_pids:
            match = (pid[b] == p) & valid[b]  # (V, N)
            has_match = match.any(dim=1)  # (V,)

            row = torch.full((1 + V,), N, dtype=torch.long, device=device)
            row[0] = b
            # argmax is safe because we only read it where has_match==True
            row[1:][has_match] = match.int().argmax(dim=1)[has_match]
            gt_rows.append(row)

    # Valid detections with pid < 0 (unmatched) are not emitted as positive GT
    # rows. In training they are filtered out upstream (create_instances zeroes
    # their score), so they never reach the graph here; supervising them as
    # one-view positives would bias the diffusion toward dustbin-heavy edges. They
    # appear as candidate negatives only at inference.

    if not gt_rows:
        return torch.zeros((0, V + 1), dtype=torch.long, device=device)

    return torch.cat([r.unsqueeze(0) for r in gt_rows], dim=0)


def match_gt_to_edges(
    gt_correspondences: torch.Tensor,  # (K, 1+V)
    edge_tuples: torch.Tensor,  # (M, 1+V)
) -> torch.Tensor:
    """
    Match GT correspondences to graph edges.

    Both inputs use the same format: [batch_idx, n_0, ..., n_{V-1}]
    where N is the dustbin value for missing views.

    Returns:
        E0: (M,) binary vector, 1.0 where edge matches a GT correspondence
    """
    M = edge_tuples.shape[0]
    device = edge_tuples.device

    if M == 0 or gt_correspondences.numel() == 0:
        return torch.zeros(M, device=device, dtype=torch.float32)

    # Concatenate and assign an integer "row id" to every unique tuple
    all_rows = torch.cat([edge_tuples, gt_correspondences], dim=0)  # (M+K, 1+V)

    # uniq_rows are the unique tuples; inv maps each row -> unique index
    _, inv = torch.unique(all_rows, dim=0, return_inverse=True)

    edge_ids = inv[:M]
    gt_ids = inv[M:]

    # edges that have an id appearing in gt_ids are positives
    pos = torch.isin(edge_ids, gt_ids)

    return pos.to(torch.float32)
