"""Per-dataset specs (hardcoded sequences, cameras, image paths, calibration loaders).

Each module under this package exposes a `SPEC` instance describing one dataset.
`get_spec(name)` returns the spec for the given dataset name.
"""

from preparation.utils.datasets import campus, mm_or, panoptic, shelf
from preparation.utils.datasets.base import DatasetSpec

SPECS: dict[str, DatasetSpec] = {
    "panoptic": panoptic.SPEC,
    "shelf": shelf.SPEC,
    "campus": campus.SPEC,
    "mm_or": mm_or.SPEC,
}


def get_spec(name: str) -> DatasetSpec:
    if name not in SPECS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(SPECS)}.")
    return SPECS[name]


__all__ = ["DatasetSpec", "SPECS", "get_spec"]
