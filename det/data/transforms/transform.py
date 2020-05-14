# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from typing import Optional, Tuple

import cv2
import numpy as np
from foundation.registry import Registry
from foundation.transforms import (
    CV2_INTER_CODES,
    HFlipTransform,
    NoOpTransform,
    ResizeTransform,
    Transform,
    is_numpy,
    is_numpy_coords,
    is_numpy_image,
    is_numpy_segmentation,
)
from PIL import Image

__all__ = [
    'TransformRegistry',
    'ExtentTransform',
    'RotationTransform',
]

PIL_INTER_CODES = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanzcos': Image.LANCZOS,
}


class TransformRegistry(Registry):
    """Registry of transforms."""
    pass


# TODO: register more Transform if needed


@TransformRegistry.register('ExtentTransform')
class ExtentTransform(Transform):
    """Extracting a sub-region from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(
        self,
        src_rect: Tuple[int, int, int, int],
        output_size: Tuple[int, int],
        interp: str = 'bilinear',
        fill: int = 0,
    ) -> None:
        """
        Args:
            src_rect: Source coordinates in (x0, y0, x1, y1).
            output_size: Expected image size (h, w).
            interp: PIL interpolation methods. See :const:`PIL_INTER_CODES` for all options.
            fill: Fill color used when src_rect extends outside image.
        """
        self.src_rect = src_rect
        self.output_size = output_size
        self.interp = interp
        self.fill = fill

    def apply_image(self, image: np.ndarray, interp: Optional[str] = None) -> np.ndarray:
        if not is_numpy(image):
            raise TypeError('image should be np.ndarray. Got {}'.format(type(image)))
        if not is_numpy_image(image):
            raise ValueError('image should be 2D/3D. Got {}D'.format(image.ndim))

        h, w = self.output_size
        interp_method = interp if interp is not None else self.interp
        ret = Image.fromarray(image).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=PIL_INTER_CODES[interp_method],
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        if not is_numpy(coords):
            raise TypeError('coords should be np.ndarray. Got {}'.format(type(coords)))
        if not is_numpy_coords(coords):
            raise ValueError('coords should be of shape Nx2. Got {}'.format(coords.shape))

        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if not is_numpy(segmentation):
            raise TypeError('segmentation should be np.ndarray. Got {}'.format(type(segmentation)))
        if not is_numpy_segmentation(segmentation):
            raise ValueError('segmentation should be 2D. Got {}D'.format(segmentation.ndim))

        return self.apply_image(segmentation, interp='nearest')


@TransformRegistry.register('RotationTransform')
class RotationTransform(Transform):
    """Rotating the image with given number of degrees counter clockwise around its center."""

    def __init__(
        self,
        h: int,
        w: int,
        angle: float,
        expand: bool = True,
        center: Optional[Tuple[int, int]] = None,
        interp: str = 'bilinear',
    ) -> None:
        """
        Args:
            h: Original image height.
            w: Original image width.
            angle: Degrees for rotation
            expand: Choose if the image should be resized to fit the whole rotated image (default),
                or simply cropped.
            center: Coordinates of the rotation center (w, h). If left to None, the center will be
                fit to the center of each image center has no effect if expand=True because it only
                affects shifting.
            interp: cv2 interpolation method. See :const:`CV2_INTER_CODES` for all options.
        """
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center

        abs_cos, abs_sin = abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle)))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint([h * abs_sin + w * abs_cos,
                                        h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.h = h
        self.w = w
        self.angle = angle
        self.expand = expand
        self.center = center
        self.interp = interp

        self.bound_w = bound_w
        self.bound_h = bound_h
        self.image_center = image_center

        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, image: np.ndarray, interp: Optional[str] = None) -> np.ndarray:
        if not is_numpy(image):
            raise TypeError('image should be np.ndarray. Got {}'.format(type(image)))
        if not is_numpy_image(image):
            raise ValueError('image should be 2D/3D. Got {}D'.format(image.ndim))

        h, w = image.shape[:2]
        assert self.h == h and self.w == w, \
            'Input size mismatch h w {}:{} -> {}:{}'.format(self.h, self.w, h, w)

        if len(image) == 0 or self.angle % 360 == 0:
            return image

        interp_method = interp if interp is not None else self.interp
        return cv2.warpAffine(
            image,
            self.rm_image, (self.bound_w, self.bound_h),
            flags=CV2_INTER_CODES[interp_method]
        )

    def apply_coords(self, coords):
        if not is_numpy(coords):
            raise TypeError('coords should be np.ndarray. Got {}'.format(type(coords)))
        if not is_numpy_coords(coords):
            raise ValueError('coords should be of shape Nx2. Got {}'.format(coords.shape))

        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if not is_numpy(segmentation):
            raise TypeError('segmentation should be np.ndarray. Got {}'.format(type(segmentation)))
        if not is_numpy_segmentation(segmentation):
            raise ValueError('segmentation should be 2D. Got {}D'.format(segmentation.ndim))

        return self.apply_image(segmentation, interp='nearest')

    def create_rotation_matrix(self, offset: float = 0.) -> np.ndarray:
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm


def HFlip_rotated_box(transform: HFlipTransform, rotated_boxes: np.ndarray) -> np.ndarray:
    """Applies the horizontal flip transform on rotated boxes.

    Args:
        transform: :class:`HFlipTransform` instance.
        rotated_boxes: Floating point array of shape Nx5 of
            (x_center, y_center, width, height, angle_degrees) format in absolute coordinates.

    Returns:
        Rotated boxes.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform: ResizeTransform, rotated_boxes: np.ndarray) -> np.ndarray:
    """Apples the resizing transform on rotated boxes.

    For details of how these (approximation) formulas are derived, please refer to
    :meth:`RotatedBoxes.scale`.

    Args:
        transform: :class:`ResizeTransform` instance.
        rotated_boxes: Floating point array of shape Nx5 of
            (x_center, y_center, width, height, angle_degrees) format in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


HFlipTransform.register_type('rotated_box', HFlip_rotated_box)
NoOpTransform.register_type('rotated_box', lambda t, x: x)
ResizeTransform.register_type('rotated_box', Resize_rotated_box)
