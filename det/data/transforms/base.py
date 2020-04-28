from __future__ import absolute_import, division, print_function

from foundation.registry import Registry
from foundation.transforms import Transform


class TransformRegistry(Registry):
    """Registry of transforms."""
    pass


class WrappedTransform(object):
    """Transform which takes a dataset dict in Detectron2 Dataset format and map it."""

    def get_transform(self) -> Transform:
        raise NotImplementedError

    def __call__(self, image, annotations):
        transform = self.get_transform()

        image = transform.apply_image(image)
        return image

    def __repr__(self):
        """Nice """
        return 'test'
