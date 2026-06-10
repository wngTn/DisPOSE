from torch.utils.data import default_collate


def custom_collate(batch: list[dict], non_collatable_keys: list[str] | None = None) -> dict:
    """
    Custom collate function that handles keys that can't be processed by default_collate.

    Args:
        batch (list[dict]): A list of dictionaries, where each dictionary
                             represents a single sample.
        non_collatable_keys (list[str] | None): keys excluded from default collation and
                                                kept as plain lists. Defaults to ["img_paths"].

    Returns:
        dict: A dictionary representing the collated batch.
    """
    if non_collatable_keys is None:
        non_collatable_keys = ["img_paths"]

    # Extract non-collatable items
    extracted_items = {}
    for key in non_collatable_keys:
        extracted_items[key] = [sample.pop(key, None) for sample in batch]

    # Use default collate for remaining items
    collated_batch = default_collate(batch)

    # Add back the non-collatable items
    collated_batch.update(extracted_items)

    return collated_batch
