from __future__ import absolute_import, division, print_function

from typing import Any, Optional, Union

import numpy as np
from foundation.transforms import HFlipTransform, NoOpTransform, Transform, VFlipTransform

from .base import TransformGen, TransformGenRegistry


@TransformGenRegistry.register('RandomApply')
class RandomApply(TransformGen):
    """Randomly apply the wrapper transformation with a given probability."""

    def __init__(self, transform: Union[Transform, TransformGen], prob: float = 0.5) -> None:
        """
        Args:
            transform: The transform to be wrapped by the `RandomApply`. The `transform` can either
                be a `Transform` or `TransformGen` instance.
            prob: Probability between 0.0 and 1.0 that the wrapper transformation is applied.
        """
        super(RandomApply, self).__init__()

        if not isinstance(transform, (Transform, TransformGen)):
            raise TypeError(
                'transform should be Transform or TransformGen. Got {}'.format(type(transform))
            )

        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must be between 0.0 and 1.0. Got {}'.format(prob))

        self._transform = transform
        self._prob = prob

    def get_transform(self, image: np.ndarray, annotations: Optional[Any] = None) -> Transform:
        do = self._rand_range() < self._prob
        if do:
            if isinstance(self._transform, TransformGen):
                return self._transform.get_transform(image, annotations)
            else:
                return self._transform
        else:
            return NoOpTransform()


@TransformGenRegistry.register('RandomHFlip')
class RandomHFlip(TransformGen):
    """Flips the image horizontally with the given probability."""

    def __init__(self, prob: float = 0.5) -> None:
        super(RandomHFlip, self).__init__()

        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must be between 0.0 and 1.0. Got {}'.format(prob))

        self._prob = prob

    def get_transform(self, image: np.ndarray, annotations: Optional[Any] = None) -> Transform:
        h, w = image.shape[:2]
        do = self._rand_range() < self._prob
        if do:
            return HFlipTransform(w)
        else:
            return NoOpTransform()


@TransformGenRegistry.register('RandomVFlip')
class RandomVFlip(TransformGen):
    """Flips the image vertically with the given probability."""

    def __init__(self, prob: float = 0.5) -> None:
        super(RandomVFlip, self).__init__()

        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must be between 0.0 and 1.0. Got {}'.format(prob))

        self._prob = prob

    def get_transform(self, image: np.ndarray, annotations: Optional[Any] = None) -> Transform:
        h, w = image.shape[:2]
        do = self._rand_range() < self._prob
        if do:
            return VFlipTransform(h)
        else:
            return NoOpTransform()


# class Resize(TransformGen):
#     """ Resize image to a target size"""
#
#     def __init__(self, shape, interp=Image.BILINEAR):
#         """
#         Args:
#             shape: (h, w) tuple or a int
#             interp: PIL interpolation method
#         """
#         if isinstance(shape, int):
#             shape = (shape, shape)
#         shape = tuple(shape)
#         self._init(locals())
#
#     def get_transform(self, img):
#         return ResizeTransform(
#             img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
#         )
#
#
# class ResizeShortestEdge(TransformGen):
#     """
#     Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
#     If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
#     """
#
#     def __init__(
#         self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
#     ):
#         """
#         Args:
#             short_edge_length (list[int]): If ``sample_style=="range"``,
#                 a [min, max] interval from which to sample the shortest edge length.
#                 If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
#             max_size (int): maximum allowed longest edge length.
#             sample_style (str): either "range" or "choice".
#         """
#         super().__init__()
#         assert sample_style in ["range", "choice"], sample_style
#
#         self.is_range = sample_style == "range"
#         if isinstance(short_edge_length, int):
#             short_edge_length = (short_edge_length, short_edge_length)
#         self._init(locals())
#
#     def get_transform(self, img):
#         h, w = img.shape[:2]
#
#         if self.is_range:
#             size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
#         else:
#             size = np.random.choice(self.short_edge_length)
#         if size == 0:
#             return NoOpTransform()
#
#         scale = size * 1.0 / min(h, w)
#         if h < w:
#             newh, neww = size, scale * w
#         else:
#             newh, neww = scale * h, size
#         if max(newh, neww) > self.max_size:
#             scale = self.max_size * 1.0 / max(newh, neww)
#             newh = newh * scale
#             neww = neww * scale
#         neww = int(neww + 0.5)
#         newh = int(newh + 0.5)
#         return ResizeTransform(h, w, newh, neww, self.interp)
#
#
# class RandomRotation(TransformGen):
#     """
#     This method returns a copy of this image, rotated the given
#     number of degrees counter clockwise around the given center.
#     """
#
#     def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
#         """
#         Args:
#             angle (list[float]): If ``sample_style=="range"``,
#                 a [min, max] interval from which to sample the angle (in degrees).
#                 If ``sample_style=="choice"``, a list of angles to sample from
#             expand (bool): choose if the image should be resized to fit the whole
#                 rotated image (default), or simply cropped
#             center (list[[float, float]]):  If ``sample_style=="range"``,
#                 a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
#                 [0, 0] being the top left of the image and [1, 1] the bottom right.
#                 If ``sample_style=="choice"``, a list of centers to sample from
#                 Default: None, which means that the center of rotation is the center of the image
#                 center has no effect if expand=True because it only affects shifting
#         """
#         super().__init__()
#         assert sample_style in ["range", "choice"], sample_style
#         self.is_range = sample_style == "range"
#         if isinstance(angle, (float, int)):
#             angle = (angle, angle)
#         if center is not None and isinstance(center[0], (float, int)):
#             center = (center, center)
#         self._init(locals())
#
#     def get_transform(self, img):
#         h, w = img.shape[:2]
#         center = None
#         if self.is_range:
#             angle = np.random.uniform(self.angle[0], self.angle[1])
#             if self.center is not None:
#                 center = (
#                     np.random.uniform(self.center[0][0], self.center[1][0]),
#                     np.random.uniform(self.center[0][1], self.center[1][1]),
#                 )
#         else:
#             angle = np.random.choice(self.angle)
#             if self.center is not None:
#                 center = np.random.choice(self.center)
#
#         if center is not None:
#             center = (w * center[0], h * center[1])  # Convert to absolute coordinates
#
#         return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)
#
#
# class RandomCrop(TransformGen):
#     """
#     Randomly crop a subimage out of an image.
#     """
#
#     def __init__(self, crop_type: str, crop_size):
#         """
#         Args:
#             crop_type (str): one of "relative_range", "relative", "absolute".
#                 See `config/defaults.py` for explanation.
#             crop_size (tuple[float]): the relative ratio or absolute pixels of
#                 height and width
#         """
#         super().__init__()
#         assert crop_type in ["relative_range", "relative", "absolute"]
#         self._init(locals())
#
#     def get_transform(self, img):
#         h, w = img.shape[:2]
#         croph, cropw = self.get_crop_size((h, w))
#         assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
#         h0 = np.random.randint(h - croph + 1)
#         w0 = np.random.randint(w - cropw + 1)
#         return CropTransform(w0, h0, cropw, croph)
#
#     def get_crop_size(self, image_size):
#         """
#         Args:
#             image_size (tuple): height, width
#
#         Returns:
#             crop_size (tuple): height, width in absolute pixels
#         """
#         h, w = image_size
#         if self.crop_type == "relative":
#             ch, cw = self.crop_size
#             return int(h * ch + 0.5), int(w * cw + 0.5)
#         elif self.crop_type == "relative_range":
#             crop_size = np.asarray(self.crop_size, dtype=np.float32)
#             ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
#             return int(h * ch + 0.5), int(w * cw + 0.5)
#         elif self.crop_type == "absolute":
#             return (min(self.crop_size[0], h), min(self.crop_size[1], w))
#         else:
#             NotImplementedError("Unknown crop type {}".format(self.crop_type))
#
#
# class RandomExtent(TransformGen):
#     """
#     Outputs an image by cropping a random "subrect" of the source image.
#
#     The subrect can be parameterized to include pixels outside the source image,
#     in which case they will be set to zeros (i.e. black). The size of the output
#     image will vary with the size of the random subrect.
#     """
#
#     def __init__(self, scale_range, shift_range):
#         """
#         Args:
#             output_size (h, w): Dimensions of output image
#             scale_range (l, h): Range of input-to-output size scaling factor
#             shift_range (x, y): Range of shifts of the cropped subrect. The rect
#                 is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
#                 where (w, h) is the (width, height) of the input image. Set each
#                 component to zero to crop at the image's center.
#         """
#         super().__init__()
#         self._init(locals())
#
#     def get_transform(self, img):
#         img_h, img_w = img.shape[:2]
#
#         # Initialize src_rect to fit the input image.
#         src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])
#
#         # Apply a random scaling to the src_rect.
#         src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])
#
#         # Apply a random shift to the coordinates origin.
#         src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
#         src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)
#
#         # Map src_rect coordinates into image coordinates (center at corner).
#         src_rect[0::2] += 0.5 * img_w
#         src_rect[1::2] += 0.5 * img_h
#
#         return ExtentTransform(
#             src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
#             output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
#         )
#
#
# class RandomContrast(TransformGen):
#     """
#     Randomly transforms image contrast.
#
#     Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
#     - intensity < 1 will reduce contrast
#     - intensity = 1 will preserve the input image
#     - intensity > 1 will increase contrast
#
#     See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
#     """
#
#     def __init__(self, intensity_min, intensity_max):
#         """
#         Args:
#             intensity_min (float): Minimum augmentation
#             intensity_max (float): Maximum augmentation
#         """
#         super().__init__()
#         self._init(locals())
#
#     def get_transform(self, img):
#         w = np.random.uniform(self.intensity_min, self.intensity_max)
#         return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)
#
#
# class RandomBrightness(TransformGen):
#     """
#     Randomly transforms image brightness.
#
#     Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
#     - intensity < 1 will reduce brightness
#     - intensity = 1 will preserve the input image
#     - intensity > 1 will increase brightness
#
#     See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
#     """
#
#     def __init__(self, intensity_min, intensity_max):
#         """
#         Args:
#             intensity_min (float): Minimum augmentation
#             intensity_max (float): Maximum augmentation
#         """
#         super().__init__()
#         self._init(locals())
#
#     def get_transform(self, img):
#         w = np.random.uniform(self.intensity_min, self.intensity_max)
#         return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)
#
#
# class RandomSaturation(TransformGen):
#     """
#     Randomly transforms image saturation.
#
#     Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
#     - intensity < 1 will reduce saturation (make the image more grayscale)
#     - intensity = 1 will preserve the input image
#     - intensity > 1 will increase saturation
#
#     See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
#     """
#
#     def __init__(self, intensity_min, intensity_max):
#         """
#         Args:
#             intensity_min (float): Minimum augmentation (1 preserves input).
#             intensity_max (float): Maximum augmentation (1 preserves input).
#         """
#         super().__init__()
#         self._init(locals())
#
#     def get_transform(self, img):
#         assert img.shape[-1] == 3, "Saturation only works on RGB images"
#         w = np.random.uniform(self.intensity_min, self.intensity_max)
#         grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
#         return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)
#
#
# class RandomLighting(TransformGen):
#     """
#     Randomly transforms image color using fixed PCA over ImageNet.
#
#     The degree of color jittering is randomly sampled via a normal distribution,
#     with standard deviation given by the scale parameter.
#     """
#
#     def __init__(self, scale):
#         """
#         Args:
#             scale (float): Standard deviation of principal component weighting.
#         """
#         super().__init__()
#         self._init(locals())
#         self.eigen_vecs = np.array(
#             [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
#         )
#         self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])
#
#     def get_transform(self, img):
#         assert img.shape[-1] == 3, "Saturation only works on RGB images"
#         weights = np.random.normal(scale=self.scale, size=3)
#         return BlendTransform(
#             src_image=self.eigen_vecs.dot(weights * self.eigen_vals),
#             src_weight=1.0,
#             dst_weight=1.0
#         )
