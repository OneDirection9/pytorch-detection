from __future__ import absolute_import, division, print_function

import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional

from foundation.registry import Registry

__all__ = ['PipelineRegistry', 'Pipeline', 'Compose']


class PipelineRegistry(Registry):
    """Registry of pipelines."""
    pass


class Pipeline(object, metaclass=ABCMeta):
    """Base pipeline class.

    A pipeline takes a single example produced by the :class:`VisionDataset` as input and returns
    processed example or None. When returning None, the example should be ignored.

    Typical pipeline use cases are filtering out invalid annotations, converting loaded examples to
    the format accepted by downstream modules, and so on, which are only need to do once during the
    whole workflow.

    Notes:
        Don't do memory heavy work in pipelines, such as loading images. Because the examples
        returned by pipelines should be passed to :class:`DatasetFromList` to get a PyTorch format
        class. If loaded images, the memory cost is expensive.
    """

    def __init__(self):
        """Rewrites it to avoid raise AssertionError in :meth:`__repr__` due to *args, **kwargs."""
        pass

    @abstractmethod
    def __call__(self, example: Any) -> Optional[Any]:
        pass

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


class Compose(object):
    """A class that composes several pipelines together."""

    def __init__(self, pipelines: List[Pipeline]) -> None:
        """
        Args:
            pipelines: List of pipelines which are executed one by one.
        """
        for ppl in pipelines:
            if not isinstance(ppl, Pipeline):
                raise TypeError('Expected Pipeline. Got {}'.format(type(ppl)))

        self._pipelines = pipelines

    def __call__(self, example: Any) -> Optional[Any]:
        for ppl in self._pipelines:
            example = ppl(example)
            if example is None:
                return None
        return example

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for ppl in self._pipelines:
            format_string += '\n'
            format_string += '    {0}'.format(ppl)
        format_string += '\n)'
        return format_string

    __str__ = __repr__
