import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from torch import Tensor


class BaseLosses(nn.Module):
    def __init__(self, cfg: DictConfig, **kwargs):
        """
        Loss-aggregation registry. Each entry is one of:

            "<name>":              {"lambda": <w>, "type": Callable}     # single loss
            "<stage_template>":    {"lambda": <w>, "n_stages": N, "type": Callable}    # uniform per stage
            "<stage_template>":    {"lambdas": [w0, ..., wN-1], "type": Callable}     # varying per stage

        Templates with ``n_stages`` or ``lambdas`` are expanded internally to
        ``"<stage_template>_0"``, ..., ``"<stage_template>_{N-1}"`` so the rest of the
        machinery (and the per-stage lookup in :class:`src.losses.loss.Loss`) sees a
        flat dict — keeps the runtime simple while letting configs be terse.
        """
        super().__init__()

        cfg = self._expand_stage_templates(cfg)

        losses = list(cfg.keys())
        self._lambda_weights = {k: v["lambda"] for k, v in cfg.items()}

        if "total" not in losses:
            losses.append("total")

        for loss in losses:
            self.register_buffer(str(loss), torch.tensor(0.0), persistent=False)

        self.register_buffer("count", torch.tensor(0.0), persistent=False)
        self.losses = losses

        self._losses_func = nn.ModuleDict()
        for loss in losses:
            if loss == "total":
                continue
            self._losses_func[loss] = cfg[loss]["type"]

    @staticmethod
    def _expand_stage_templates(cfg: DictConfig) -> dict:
        """Expand ``n_stages`` / ``lambdas`` template entries into per-stage entries.

        A template entry ``"X"`` with ``n_stages: N`` becomes ``"X_0", ..., "X_{N-1}"``,
        each pointing at the same loss instance config (``type``) and inheriting
        ``lambda`` (uniform across stages) or ``lambdas[i]`` (per-stage). Non-template
        entries pass through unchanged.
        """
        expanded: dict = {}
        for key, val in cfg.items():
            has_n_stages = "n_stages" in val
            has_lambdas = "lambdas" in val
            if not (has_n_stages or has_lambdas):
                expanded[key] = val
                continue
            if has_lambdas:
                lambdas = list(val["lambdas"])
                n_stages = len(lambdas)
                if has_n_stages and int(val["n_stages"]) != n_stages:
                    raise ValueError(
                        f"Loss '{key}': len(lambdas)={n_stages} disagrees with "
                        f"n_stages={int(val['n_stages'])}."
                    )
            else:
                n_stages = int(val["n_stages"])
                if "lambda" not in val:
                    raise ValueError(f"Loss '{key}' has n_stages but no 'lambda' or 'lambdas'.")
                lambdas = [val["lambda"]] * n_stages
            for i, lam in enumerate(lambdas):
                expanded[f"{key}_{i}"] = {"lambda": lam, "type": val["type"]}
        return expanded

    def _update_loss(self, loss: str, *args, **kwargs) -> Tensor:
        """
        Update the accumulated loss and return the weighted loss.

        Args:
            loss: Name of the loss to update
            outputs: Model outputs
            inputs: Ground truth inputs

        Returns:
            Weighted loss value for backpropagation
        """
        weight = self._lambda_weights[loss]
        if weight == 0:
            return self.total.new_tensor(0.0)

        # Calculate the loss
        val = self._losses_func[loss](*args, **kwargs)

        getattr(self, loss).add_(val.detach())

        # Return weighted loss for backpropagation
        weighted_loss = weight * val
        return weighted_loss

    def reset(self):
        """Reset the losses to 0."""
        for loss in self.losses:
            setattr(self, str(loss), torch.tensor(0.0, device=getattr(self, str(loss)).device))
        setattr(self, "count", torch.tensor(0.0, device=getattr(self, "count").device))

    def compute(self):
        """Compute the losses and return a dictionary with the losses.

        Returns a stable key set across all ranks: losses that received no
        updates in this window (count == 0 → NaN) are reported as 0.0 rather
        than filtered out. DDP requires the keys passed to ``self.log_dict``
        to be identical on every rank — otherwise per-key metric reduction
        collectives mismatch and NCCL times out.
        """
        count = self.count
        log_dict: dict[str, torch.Tensor] = {}
        for loss in self.losses:
            accum = getattr(self, str(loss))
            value = (accum / count).detach() if count > 0 else accum.detach().clone()
            log_dict[str(loss)] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        # Reset the losses
        self.reset()
        return log_dict
