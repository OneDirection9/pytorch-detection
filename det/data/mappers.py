from __future__ import absolute_import, division, print_function

from functools import partial
from typing import Any, Dict, Optional

from foundation.registry import Registry

from . import transforms as T, utils


class MapperRegistry(Registry):
    """Registry of mappers."""
    pass


@MapperRegistry.register('ImageLoader')
class ImageLoader(object):

    def __init__(self, image_format='BGR'):
        self.image_format = image_format

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image = utils.read_image(example['file_name'], self.image_format)
        utils.check_image_size(example, image)
        example['image'] = image
        return example


class TransformApplier(object):

    def __init__(self, transform_cls, *args, **kwargs):
        self.transform = transform_cls(*args, **kwargs)

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if isinstance(self.transform, T.TransformGen):
            transform = self.transform.get_transform(example['image'])
        else:
            transform = self.transform
        example['image'] = transform.apply_image(example['image'])
        image_shape = example['']
        return image_shape

    def __repr__(self):
        return repr(self.transform)


MapperRegistry.register('hflip')(partial(TransformApplier, T.RandomHFlip))
