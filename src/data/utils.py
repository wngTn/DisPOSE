import numpy as np


def get_scale(image_size, resized_size):
    """Calculates scaling factors based on original image dimensions and a target resized size.

    This function determines padded dimensions (`w_pad`, `h_pad`) that maintain the
    original image's aspect ratio relative to the `resized_size`, and then normalizes
    these padded dimensions by a base value of 200.0. This is typically used to
    compute scale factors for image processing operations like resizing or padding
    to fit a specific model input.

    Args:
        image_size (tuple): A tuple `(width, height)` representing the original
                            dimensions of the image.
        resized_size (tuple): A tuple `(resized_width, resized_height)` representing
                              the target dimensions or a reference size for scaling.

    Returns:
        numpy.ndarray: A 2-element NumPy array `[scale_x, scale_y]` of type
                       `np.float32`, where `scale_x = w_pad / 200.0` and
                       `scale_y = h_pad / 200.0`.
    """
    w, h = image_size
    w_resized, h_resized = resized_size
    if w / w_resized < h / h_resized:
        w_pad = h / h_resized * w_resized
        h_pad = h
    else:
        w_pad = w
        h_pad = w / w_resized * h_resized
    scale = np.array([w_pad / 200.0, h_pad / 200.0], dtype=np.float32)

    return scale
