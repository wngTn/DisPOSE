import torch
import torch.distributed
import torch.nn as nn


class NoOp:
    """Null object whose attribute access returns a no-op callable.

    Used as a stand-in for an optional logger/handler so call sites can invoke
    ``obj.anything(...)`` unconditionally without a None check.
    """

    def __init__(self):
        self.log_iter = 1

    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op


def move_to_device(obj, device):
    """
    Recursively moves torch.Tensors within nested lists and dictionaries to the specified device.

    Args:
        obj (any): The object to traverse. Can be a torch.Tensor, list, dictionary,
                   or other data type.
        device (torch.device or str): The target device (e.g., 'cuda', 'cpu', 'cuda:0').

    Returns:
        any: The object with all torch.Tensors moved to the specified device.
    """
    if torch.is_tensor(obj):
        return obj.detach().to(device)
    elif isinstance(obj, nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(elem, device) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(elem, device) for elem in obj)
    else:
        return obj


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def np2ts(x, **kwargs):
    return torch.tensor(x, **kwargs)


def get_rank():
    if not torch.distributed.is_available():  # type: ignore
        return 0
    if not torch.distributed.is_initialized():  # type: ignore
        return 0
    return torch.distributed.get_rank()  # type: ignore
