# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import inspect
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from foundation.transforms import Transform, TransformList

__all__ = [
    'Augmentation',
    'AugInput',
    'StandardAugInput',
]
"""
Overview of the augmentation system:

We have a design goal that aims at allowing:
    (1) Arbitrary structures of input data (e.g. list[list[boxes]], dict[str, boxes],
        multiple semantic segmentations for each image, etc) and arbitrary new data types
        (rotated boxes, 3D meshes, densepose, etc)
    (2) A list of augmentation to be applied sequentially

`Augmentation` defines policies to create deterministic transforms from input data.
An augmentation policy may need to access arbitrary input data, so it declares the input
data needed, to be provided by users when calling its `get_transform` method.

`Augmentation` is not able to apply transforms to data: data associated with
one sample may be much more than what `Augmentation` gets. For example, most
augmentation policies only need an image, but the actual input samples can be
much more complicated.

`AugInput` manages all inputs needed by `Augmentation` and implements the logic
to apply a sequence of augmentation. It has to define how the inputs are transformed,
because arguments needed by one `Augmentation` needs to be transformed to become arguments
of the next `Augmentation` in the sequence.

`AugInput` does not need to contain all input data, because most augmentation policies
only need very few fields (e.g., most only need "image"). We provide `StandardAugInput`
that only contains "images", "boxes", "sem_seg", that are enough to create transforms
for most cases. In this way, users keep the responsibility to apply transforms to other
potentially new data types and structures, e.g. keypoints, proposals boxes.

To extend the system, one can do:
1. To add a new augmentation policy that only needs to use standard inputs
   ("image", "boxes", "sem_seg"), writing a subclass of `Augmentation` is sufficient.
2. To use new data types or custom data structures, `StandardAugInput` can still be used as long
   as the new data types or custom data structures are not needed by any augmentation policy.
   The new data types or data structures can be transformed using the
   transforms returned by `AugInput.apply_augmentations`.
3. To add new augmentation policies that need new data types or data structures, in addition to
   implementing new `Augmentation`, a new `AugInput` is needed as well.
"""


class Augmentation(metaclass=ABCMeta):
    """
    Augmentation defines policies/strategies to generate :class:`Transform` from data.
    It is often used for pre-processing of input data. A policy typically contains
    randomness, but it can also choose to deterministically generate a :class:`Transform`.

    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method with the :attr:`input_args` attribute.
    When called with the positional arguments defined by the :attr:`input_args`,
    the :meth:`get_transform` method executes the policy.

    Examples:
    ::
        # if a policy needs to know both image and semantic segmentation
        assert aug.input_args == ("image", "sem_seg")
        tfm: Transform = aug.get_transform(image, sem_seg)
        new_image = tfm.apply_image(image)

    To implement a custom :class:`Augmentation`, define its :attr:`input_args` and
    implement :meth:`get_transform`.

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to apply the actual transform to those data.
    """

    input_args: Tuple[str] = ('image',)
    """
    Attribute of class instances that defines the argument(s) needed by
    :meth:`get_transform`. Default to only "image", because most policies only
    require knowing the image in order to determine the transform.

    Users can freely define arbitrary new args and their types in custom
    :class:`Augmentation`. In detectron2 we use the following convention:

    * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
      floating point in range [0, 1] or [0, 255].
    * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
      of N instances. Each is in XYXY format in unit of absolute coordinates.
    * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.

    We do not specify convention for other types and do not include builtin
    :class:`Augmentation` that uses other types in detectron2.
    """

    def __init__(self) -> None:
        """Rewrites it to avoid raise AssertionError in :meth:`__repr__` due to *args, **kwargs."""
        pass

    # NOTE: in the future, can allow it to return list[Augmentation],
    # to delegate augmentation to others
    @abstractmethod
    def get_transform(self, *args) -> Transform:
        """
        Execute the policy to use input data to create transform(s).

        Args:
            arguments must follow what's defined in :attr:`input_args`.

        Returns:
            Return a :class:`Transform` instance, which is the transform to apply to inputs.
        """
        pass

    @staticmethod
    def _rand_range(
        low=1.0, high: Optional[float] = None, size: Optional[int] = None
    ) -> Union[np.ndarray, float]:  # yapf:disable
        """Uniforms float random number between low and high.

        See :func:`np.random.uniform`.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self) -> str:
        """Produces something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            items = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    'Attribute {} not found! '
                    'Default __repr__ only works if attributes match the constructor.'.format(name)
                )
                attr = getattr(self, name)
                items.append('{}={!r}'.format(name, attr))
            return '{}({})'.format(self.__class__.__name__, ', '.join(items))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


class AugInput(object):
    """A base class for anything on which a list of :class:`Augmentation` can be applied.

    This class provides input arguments for :class:`Augmentation` to use, and defines how
    to apply transforms to these data.

    An instance of this class must satisfy the following:

    * :class:`Augmentation` declares some data it needs as arguments. A :class:`AugInput`
      must provide access to these data in the form of attribute access (``getattr``).
      For example, if a :class:`Augmentation` to be applied needs "image" and "sem_seg"
      arguments, this class must have the attribute "image" and "sem_seg" whose content
      is as required by the :class:`Augmentation`s.
    * This class must have a :meth:`transform(tfm: Transform) -> None` method which
      in-place transforms all attributes stored in the class.
    """

    def transform(self, tfm: Transform) -> None:
        raise NotImplementedError

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """Applies a list of Transform/Augmentation in-place and returned the applied transform.

        Attributes of this class will be modified.

        Returns:
            TransformList:
                returns transformed inputs and the list of transforms applied.
                The TransformList can then be applied to other data associated with the inputs.
        """
        tfms = []
        for aug in augmentations:
            if isinstance(aug, Augmentation):
                args = []
                for f in aug.input_args:
                    try:
                        args.append(getattr(self, f))
                    except AttributeError:
                        raise AttributeError(
                            f"Augmentation {aug} needs '{f}', which is not an attribute of {self}!"
                        )

                tfm = aug.get_transform(*args)
                assert isinstance(tfm, Transform), \
                    f'{type(aug)}.get_transform must return an instance of Transform! ' \
                    f'Got {type(tfm)} instead.'
            else:
                tfm = aug
            self.transform(tfm)
            tfms.append(tfm)
        return TransformList(tfms)


class StandardAugInput(AugInput):
    """
    A standard implementation of :class:`AugInput` for the majority of use cases.
    This class provides the following standard attributes that are common to use by
    Augmentation (augmentation policies). These are chosen because most
    :class:`Augmentation` won't need anything more to define a augmentation policy.
    After applying augmentations to these special attributes, the returned transforms
    can then be used to transform other data structures that users have.

    Attributes:
        image (ndarray): image in HW or HWC format. The meaning of C is up to users
        boxes (ndarray or None): Nx4 boxes in XYXY_ABS mode
        sem_seg (ndarray or None): HxW semantic segmentation mask

    Examples:
    ::
        input = StandardAugInput(image, boxes=boxes)
        tfms = input.apply_augmentations(list_of_augmentations)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may require augmentation
    policies that need more inputs. An algorithm may need to transform inputs
    in a way different from the standard approach defined in this class. In those
    situations, users can implement new subclasses of :class:`AugInput` with differnt
    attributes and the :meth:`transform` method.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255].
            boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
                of N instances. Each is in XYXY format in unit of absolute coordinates.
            sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.
        """
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)
