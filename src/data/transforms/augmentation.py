import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class Cutout:
    """Apply cutout augmentation with probability and magnitude control."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.round(np.linspace(0, 20, 10), 0).astype(int)

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[upper_coord[0] : lower_coord[0], upper_coord[1] : lower_coord[1], :] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        if random.random() >= self.prob:
            return pil_img

        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        size = self.magnitude_range[self.magnitude]

        if size == 0:
            return pil_img

        _, upper_coord, lower_coord = self._create_cutout_mask(img_height, img_width, num_channels, size)
        pixels = pil_img.load()
        for i in range(upper_coord[0], lower_coord[0]):
            for j in range(upper_coord[1], lower_coord[1]):
                pixels[i, j] = (125, 122, 113, 0)
        return pil_img


# Individual Augmentation Classes
class RandomSharpness:
    """Apply sharpness adjustment with random magnitude."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0.0, 0.9, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = ImageEnhance.Sharpness(img).enhance(1 + magnitude_value * random.choice([-1, 1]))
        return img


class RandomAutoContrast:
    """Apply auto contrast adjustment."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor

    def __call__(self, img):
        if random.random() < self.prob:
            img = ImageOps.autocontrast(img)
        return img


class RandomPosterize:
    """Apply posterize effect."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.round(np.linspace(8, 4, 10), 0).astype(int)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = ImageOps.posterize(img, magnitude_value)
        return img


class RandomEqualize:
    """Apply histogram equalization."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor

    def __call__(self, img):
        if random.random() < self.prob:
            img = ImageOps.equalize(img)
        return img


class RandomContrast:
    """Apply contrast adjustment with random magnitude."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0.0, 0.9, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = ImageEnhance.Contrast(img).enhance(1 + magnitude_value * random.choice([-1, 1]))
        return img


class RandomColor:
    """Apply color adjustment with random magnitude."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0.0, 0.9, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = ImageEnhance.Color(img).enhance(1 + magnitude_value * random.choice([-1, 1]))
        return img


class RandomBrightness:
    """Apply brightness adjustment with random magnitude."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0.0, 0.9, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = ImageEnhance.Brightness(img).enhance(1 + magnitude_value * random.choice([-1, 1]))
        return img


class RandomSolarize:
    """Apply solarize effect."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(256, 0, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = ImageOps.solarize(img, magnitude_value)
        return img


class RandomInvert:
    """Apply color inversion."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor

    def __call__(self, img):
        if random.random() < self.prob:
            img = ImageOps.invert(img)
        return img


class RandomRotate:
    """Apply rotation with random magnitude."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0, 30, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            rot = img.convert("RGBA").rotate(magnitude_value)
            img = Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)
        return img


class RandomShearX:
    """Apply horizontal shear transformation."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0, 0.3, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude_value * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=self.fillcolor,
            )
        return img


class RandomShearY:
    """Apply vertical shear transformation."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0, 0.3, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude_value * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=self.fillcolor,
            )
        return img


class RandomTranslateX:
    """Apply horizontal translation."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0, 150 / 331, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude_value * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=self.fillcolor,
            )
        return img


class RandomTranslateY:
    """Apply vertical translation."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.prob = prob
        self.magnitude = magnitude
        self.fillcolor = fillcolor
        self.magnitude_range = np.linspace(0, 150 / 331, 10)

    def __call__(self, img):
        if random.random() < self.prob:
            magnitude_value = self.magnitude_range[self.magnitude]
            img = img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude_value * img.size[1] * random.choice([-1, 1])),
                fillcolor=self.fillcolor,
            )
        return img


class RandomCutout:
    """Apply cutout augmentation - alias for Cutout for naming consistency."""

    def __init__(self, prob=0.5, magnitude=5, fillcolor=(128, 128, 128)):
        self.cutout = Cutout(prob=prob, magnitude=magnitude, fillcolor=fillcolor)

    def __call__(self, img):
        return self.cutout(img)


class RandAugmentation:
    """
    Flexible augmentation wrapper that works with Hydra configuration.

    This class allows you to specify a list of augmentation classes and randomly
    applies N of them per image. Works similar to RandAugment but with more flexibility.

    Args:
        augmentations: List of augmentation callables (can be partial functions from Hydra)
        num_ops: Number of augmentations to apply per image (default: 2)

    Example Hydra config:
        _target_: src.data.transforms.RandAugmentation
        num_ops: 2
        augmentations:
          - _target_: src.data.transforms.RandomSharpness
            _partial_: true
            prob: 0.5
            magnitude: 7
          - _target_: src.data.transforms.RandomColor
            _partial_: true
            prob: 0.5
            magnitude: 6
          - _target_: src.data.transforms.Cutout
            _partial_: true
            prob: 0.3
            magnitude: 5
    """

    def __init__(self, augmentations=None, num_ops=2):
        """
        Initialize RandAugmentation.

        Args:
            augmentations: List of augmentation classes/partials. If None or empty,
                          acts as identity transform (useful for validation)
            num_ops: Number of augmentations to randomly select and apply per image
        """
        self.augmentations = augmentations if augmentations is not None else []
        self.num_ops = num_ops

        # If augmentations is empty, set num_ops to 0 to avoid errors
        if len(self.augmentations) == 0:
            self.num_ops = 0

    def __call__(self, img):
        """
        Apply num_ops randomly selected augmentations to the image.

        Args:
            img: PIL Image

        Returns:
            Augmented PIL Image
        """
        if self.num_ops == 0 or len(self.augmentations) == 0:
            return img

        # Randomly select num_ops augmentations (with replacement)
        num_to_apply = min(self.num_ops, len(self.augmentations))
        selected_augs = random.choices(self.augmentations, k=num_to_apply)

        # Apply each selected augmentation
        for aug in selected_augs:
            img = aug(img)

        return img

    def __repr__(self):
        if len(self.augmentations) == 0:
            return "RandAugmentation(num_ops=0, augmentations=[])"
        return f"RandAugmentation(num_ops={self.num_ops}, augmentations={len(self.augmentations)})"
