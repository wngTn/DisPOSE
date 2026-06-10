from typing import Callable
from torch import nn


class BaseMetrics(nn.Module):
    def __init__(
        self, 
        cfg: dict[str, Callable], 
        **kwargs
    ) -> None:
        super().__init__()
        
        for metric_name, function in cfg.items():
            setattr(self, metric_name, function)
