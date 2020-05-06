from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Optional

from .base import Pipeline, PipelineRegistry


@PipelineRegistry.register('FormatConverter')
class FormatConverter(Pipeline):
    """A class that converts format acceptable by :class:`Transform`."""

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass
