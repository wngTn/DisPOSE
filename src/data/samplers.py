import random
import numpy as np
import torch
import torch.distributed
import torch.utils.data


def sync_random_sample_list(obj_list, k, common_choice=False):
    """Randomly sample k items from obj_list, broadcasting the chosen indices across all distributed ranks so every process selects the same elements."""
    if common_choice:
        # Use random.choices to allow for sampling with replacement
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    elif len(obj_list) < k:
        # Fallback to sampling with replacement if k is larger than the list
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        # Sample without replacement
        idx = torch.randperm(len(obj_list))[:k]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Ensure all processes have the same indices
        if torch.cuda.is_available():
            idx = idx.cuda()
        torch.distributed.broadcast(idx, src=0)

    idx = idx.tolist()
    return [obj_list[i] for i in idx]


class InferenceSampler(torch.utils.data.Sampler):
    """
    Sampler for inference that distributes data across GPUs.
    The batch_size is specified PER GPU.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = False, num_instances: int = -1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_instances = num_instances

        # Safely get world_size and rank for distributed and non-distributed scenarios
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Determine the effective size of the dataset
        if self.num_instances == -1:
            initial_size = len(dataset)
        else:
            initial_size = min(len(dataset), self.num_instances)

        indices = list(range(initial_size))

        # Shuffle the indices if shuffle is True
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(0)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # Keep all indices (no truncation)
        self.size = len(indices)

        # Create micro-batches (last batch may be smaller)
        micro_batches = []
        for i in range(0, self.size, self.batch_size):
            micro_batches.append(indices[i : i + self.batch_size])

        # Assign micro-batches to this rank in round-robin fashion
        self.idx_batch_this_rank = micro_batches[self.rank :: self.world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.idx_batch_this_rank)


class CommonSampler:
    """
    An infinite batch sampler that provides random samples, with batch_size specified PER GPU.

    No ``__len__`` is defined because the sampler is infinite.  This makes
    Lightning treat the training dataloader as unsized, which is required for
    purely step-based training (``max_steps`` + integer ``val_check_interval``).
    """

    def __init__(self, dataset, batch_size, random=True):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, but got {batch_size}")
        self.batch_shuffle = random  # Note: `random` is a Python module, renamed to avoid conflict

        # Safely get world_size and rank
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            total_batch_size = self.batch_size * self.world_size

            # sync_random_sample_list returns identical indices on all ranks; shard per rank, then optionally shuffle the local batch.
            sample_indices = sync_random_sample_list(indices_list, total_batch_size, common_choice=True)

            my_indices = sample_indices[self.rank :: self.world_size]

            if self.batch_shuffle:
                np.random.shuffle(my_indices)

            yield my_indices
