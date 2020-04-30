from __future__ import absolute_import, division, print_function

import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from foundation.registry import Registry

__all__ = ['PipelineRegistry', 'Pipeline']


class PipelineRegistry(Registry):
    """Registry of pipelines."""
    pass


class Pipeline(object, metaclass=ABCMeta):
    """Base pipeline class.

    A pipeline takes a single example produced by the :class:`VisionDataset` as input and returns
    processed example or None. When returning None, the example should be ignored.

    Typical pipeline use cases are filtering out invalid examples, converting to the format accepted
    by downstream modules, and so on.

    Note that don't load image in pipeline step, in map_func instead. Because the examples returned
    by pipeline should be passed to :class:`DatasetFromList`.
    """

    def __init__(self):
        """Rewrites it to avoid raise AssertionError in :meth:`__repr__` due to *args, **kwargs."""
        pass

    @abstractmethod
    def __call__(self, example: Any) -> Optional[Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        """Produces something like:
        MyPipeline(field1={self._field1}, field2={self._field2})
        """
        try:
            sig = inspect.signature(self.__init__)
            items = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"

                attr_name = '_{}'.format(name)
                assert hasattr(self, attr_name), (
                    'Attribute {} not found! '
                    'Default __repr__ only works if attributes match the constructor. '
                    'The matched attribute name for parameter name `a` is `_a`'.format(attr_name)
                )
                attr = getattr(self, attr_name)
                items.append('{}={!r}'.format(name, attr))
            return '{}({})'.format(self.__class__.__name__, ', '.join(items))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__
