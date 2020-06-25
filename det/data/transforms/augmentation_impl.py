# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import sys
from typing import List, Optional, Tuple, Union

import numpy as np
from foundation.transforms import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    ResizeTransform,
    Transform,
    VFlipTransform,
)

from .augmentation import Augmentation
from .transform import ExtentTransform, RotationTransform

__all__ = [
    'RandomApply',
    'RandomHFlip',
    'RandomVFlip',
    'Resize',
    'ResizeShortestEdge',
    'RandomCrop',
    'RandomContrast',
    'RandomBrightness',
    'RandomSaturation',
    'RandomLighting',
    'RandomRotation',
    'RandomExtent',
]


class RandomApply(Augmentation):
    """Applying the wrapper transformation with a given probability randomly."""

    def __init__(self, transform: Union[Augmentation, Transform], prob: float = 0.5) -> None:
        """
        Args:
            transform: The transform to be wrapped by the :class:`RandomApply`. The `transform` can
                either be a :class:`Augmentation` or :class:`Transform` instance.
            prob: The probability between 0.0 and 1.0 that the wrapper transformation is applied.
        """
        super(RandomApply, self).__init__()

        if not isinstance(transform, (Augmentation, Transform)):
            raise TypeError(
                'transform should be Augmentation or Transform. Got {}'.format(type(transform))
            )

        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must be between 0.0 and 1.0. Got {}'.format(prob))

        self.transform = transform
        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob
        if do:
            if isinstance(self.transform, Augmentation):
                return self.transform.get_transform(image)
            else:
                return self.transform
        else:
            return NoOpTransform()


class RandomHFlip(Augmentation):
    """Flipping the image horizontally with the given probability."""

    def __init__(self, prob: float = 0.5) -> None:
        """
        Args:
            prob: Probability between 0.0 and 1.0 that the horizontal flip is applied.
        """
        super(RandomHFlip, self).__init__()

        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must be between 0.0 and 1.0. Got {}'.format(prob))

        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return HFlipTransform(w)
        else:
            return NoOpTransform()


class RandomVFlip(Augmentation):
    """Flipping the image vertically with the given probability."""

    def __init__(self, prob: float = 0.5) -> None:
        """
        Args:
            prob: Probability between 0.0 and 1.0 that the vertical flip is applied.
        """
        super(RandomVFlip, self).__init__()

        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must be between 0.0 and 1.0. Got {}'.format(prob))

        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """Resizing image to a target size."""

    def __init__(self, shape: Union[Tuple[int, int], int], interp: str = 'bilinear') -> None:
        """
        Args:
            shape: (H, W) tuple or a int.
            interp: The interpolation method. See :const:`INTERP_CODES` in :module:`foundation`.
        """
        super(Resize, self).__init__()

        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = tuple(shape)
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """Scaling the shorter edge to the given size, with a limit of `max_size` on the longer edge.

    If `max_size` is reached, then downscale so that the longer edge does not exceed `max_size`.
    """

    def __init__(
        self,
        short: Union[List[int], int],
        max_size: int = sys.maxsize,
        sample_style: str = 'range',
        interp: str = 'bilinear'
    ) -> None:
        """
        Args:
            short: If `sample_style=='range'`, a [min, max] interval from which to sample the
                shortest edge length. If `sample_style=='choice'`, a list of shortest edge lengths
                to sample from.
            max_size: Maximum allowed longest edge length.
            sample_style: Either 'range' or 'choice'.
            interp: The interpolation method. See :const:`INTERP_CODES` in :module:`foundation`.
        """
        super(ResizeShortestEdge, self).__init__()

        if sample_style not in ['range', 'choice']:
            raise ValueError('sample_type should be range or choice. Got {}'.format(sample_style))

        if isinstance(short, int):
            short = (short, short)

        self.short = short
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]

        if self.sample_style == 'range':
            size = np.random.randint(self.short[0], self.short[1] + 1)
        else:
            size = np.random.choice(self.short)

        scale = size * 1.0 / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size
        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h = new_h * scale
            new_w = new_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return ResizeTransform(h, w, new_h, new_w, interp=self.interp)


class RandomCrop(Augmentation):
    """Cropping a sub-image out of an image randomly."""

    def __init__(self, crop_type: str, crop_size: Tuple[float, float]) -> None:
        """
        Args:
            crop_type: One of "relative_range", "relative", "absolute", "absolute_range". Cropping
                size calculation:
                - relative: (h * crop_size[0], w * crop_size[1]) part of an input of size (h, w).
                - relative_range: Uniformly sample relative crop size from between
                    [corp_size[0], crop_size[1]] and [1, 1] and use it as in 'relative' scenario.
                - absolute: Crop part of an input with absolute size: (crop_size[0], crop_size[1]).
            crop_size: The relative ratio or absolute pixels of height and width.
        """
        super(RandomCrop, self).__init__()

        if crop_type not in ['relative', 'relative_range', 'absolute', 'absolute_range']:
            raise ValueError(
                'crop_type should be relative, relative_range, absolute, or absolute_range. Got {}'
                .format(crop_type)
            )

        self.crop_type = crop_type
        self.crop_size = crop_size

    def get_transform(self, image: np.ndarray) -> Transform:
        """If annotations is not None, a CropTransform that the cropping region contains the center
        of a randomly selected instance. Otherwise, a region is randomly cropped out.
        """
        h, w = image.shape[:2]
        crop_h, crop_w = self.get_crop_size((h, w))
        assert h >= crop_h and w >= crop_w, 'Shape computation in {} has bugs.'.format(self)
        y1 = np.random.randint(h - crop_h + 1)
        x1 = np.random.randint(w - crop_w + 1)
        return CropTransform(x1, y1, crop_h, crop_w)

    def get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Args:
            image_size: Image height and width.

        Returns:
            crop_size: Height, width in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'relative':
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == 'absolute':
            return min(self.crop_size[0], h), min(self.crop_size[1], w)
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError('Unknown crop type {}'.format(self.crop_type))


class RandomContrast(Augmentation):
    """Transforming image contrast randomly.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min: float, intensity_max: float) -> None:
        """
        Args:
            intensity_min: Minimum augmentation.
            intensity_max: Maximum augmentation.
        """
        super().__init__()

        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def get_transform(self, image: np.ndarray) -> Transform:
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=image.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(Augmentation):
    """Transforming image brightness randomly.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min: float, intensity_max: float) -> None:
        """
        Args:
            intensity_min: Minimum augmentation.
            intensity_max: Maximum augmentation.
        """
        super(RandomBrightness, self).__init__()

        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def get_transform(self, image: np.ndarray) -> Transform:
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(Augmentation):
    """Transforming image saturation randomly.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min: float, intensity_max: float) -> None:
        """
        Args:
            intensity_min: Minimum augmentation (1 preserves input).
            intensity_max: Maximum augmentation (1 preserves input).
        """
        super(RandomSaturation, self).__init__()

        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def get_transform(self, image: np.ndarray) -> Transform:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Saturation only works on RGB images')
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = image.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(Augmentation):
    """Transforming image color using fixed PCA over ImageNet randomly.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale: float) -> None:
        """
        Args:
            scale: Standard deviation of principal component weighting.
        """
        super(RandomLighting, self).__init__()

        self.scale = scale
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, image: np.ndarray) -> Transform:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Saturation only works on RGB images')
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals),
            src_weight=1.0,
            dst_weight=1.0,
        )


class RandomRotation(Augmentation):
    """Rotating the image a few random degrees counter clockwise around the given center."""

    def __init__(
        self,
        angle: Union[List[float], float],
        expand: bool = True,
        center: Optional[List[Tuple[float, float]]] = None,
        sample_style: str = 'range',
        interp: str = 'bilinear'
    ) -> None:
        """
        Args:
            angle: If `sample_style=='range'`, a [min, max] interval from which to sample the angle
                (in degrees). If `sample_style=='choice'`, a list of angles to sample from.
            expand: Choose if the image should be resized to fit the whole rotated image (default),
                or simply cropped
            center: If `sample_style=='range'`, a [[minx, miny], [maxx, maxy]] relative interval
                from which to sample the center, [0, 0] being the top left of the image and [1, 1]
                the bottom right. If `sample_style=='choice'`, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting.
            sample_style: Either 'range' or 'choice'.
            interp: cv2 interpolation method. See :const:`CV2_INTER_CODES` for all options.
        """
        super(RandomRotation, self).__init__()

        if sample_style not in ['range', 'choice']:
            raise ValueError('sample_type should be range or choice. Got {}'.format(sample_style))
        if center is not None and (np.max(center) > 1.0 or np.min(center) < 0.0):
            raise ValueError('center should have value in range [0.0, 1.0].')

        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)

        self.angle = angle
        self.expand = expand
        self.center = center
        self.sample_style = sample_style
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]
        center = None
        if self.sample_style == 'range':
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RandomExtent(Augmentation):
    """Cropping a random 'subrect' of the source image.

    The subrect can be parameterized to include pixels outside the source image, in which case they
    will be set to zeros (i.e. black). The size of the output image will vary with the size of the
    random subrect.
    """

    def __init__(self, scale_range: Tuple[float, float], shift_range: Tuple[float, float]) -> None:
        """
        Args:
            scale_range: Range of input-to-output size scaling factor.
            shift_range: Range of shifts of the cropped subrect. The rect is shifted by
                [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)], where (w, h) is the
                (width, height) of the input image. Set each component to zero to crop at the
                image's center.
        """
        super(RandomExtent, self).__init__()

        self.scale_range = scale_range
        self.shift_range = shift_range

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * w
        src_rect[1::2] += 0.5 * h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )
