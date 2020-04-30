from __future__ import absolute_import, division, print_function

from typing import Any, List, Optional

from .base import Pipeline, PipelineRegistry

__all__ = ['Compose']


@PipelineRegistry.register('compose')
class Compose(Pipeline):
    """Composes several pipelines together."""

    def __init__(self, pipelines: List[Pipeline]) -> None:
        """
        Args:
            pipelines: List of pipelines which are executed one by one.
        """
        super().__init__()

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
