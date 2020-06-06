from __future__ import absolute_import, division, print_function

import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from foundation.nn import weight_init
from torch import nn
from torch.nn import functional as F

from det import layers
from .registry import BackboneRegistry

__all__ = [
    'BasicBlock',
    'BottleneckBlock',
    'BasicStem',
    'ResNet',
    'build_resnet_backbone',
]

logger = logging.getLogger(__name__)


class BasicBlock(layers.BaseCNNBlock):
    """The basic residual block for ResNet-18 and ResNet-34.

    The block has two 3x3 conv layers and a projection shortcut if needed defined in
    `Deep Residual Learning for Image Recognition`_.

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        norm: Union[str, Callable] = 'BN',
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride of the first conv.
            norm: Normalization for all conv layers. See :func:`get_norm` for supported format.
        """
        super(BasicBlock, self).__init__(in_channels, out_channels, stride)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = layers.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=layers.get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = layers.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=layers.get_norm(norm, out_channels),
        )

        self.conv2 = layers.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=layers.get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.caffe2_msra_init(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(layers.BaseCNNBlock):
    """The standard bottleneck block used by ResNet-50, 101 and 152.

    The block has 3 conv layers with kernels 1x1, 3x3, 1x1, and a projection shortcut if needed
    defined in `Deep Residual Learning for Image Recognition`_.

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bottleneck_channels: int,
        stride: int = 1,
        num_groups: int = 1,
        norm: Union[str, Callable] = 'BN',
        stride_in_1x1: bool = True,
        dilation: int = 1,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bottleneck_channels: Number of output channels for the 3x3 "bottleneck" conv layers.
            stride: Stride of the first conv.
            num_groups: Number of groups for the 3x3 conv layer.
            norm: Normalization for all conv layers. See :func:`get_norm` for supported format.
            stride_in_1x1: When stride>1, whether to put stride in the first 1x1 convolution or the
                bottleneck 3x3 convolution.
            dilation: The dilation rate of the 3x3 conv layer.
        """
        super(BottleneckBlock, self).__init__(in_channels, out_channels, stride)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = layers.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=layers.get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = layers.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=layers.get_norm(norm, bottleneck_channels),
        )
        self.conv2 = layers.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=layers.get_norm(norm, bottleneck_channels),
        )
        self.conv3 = layers.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=layers.get_norm(norm, out_channels)
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.caffe2_msra_init(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO: this somehow hurts performance when training GN models from scratch.
        #   Add it as an option when we need to use this code to train a backbone.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(layers.BaseCNNBlock):
    """The standard ResNet stem (layers before the first residual block)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        norm: Union[str, Callable] = 'BN'
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            norm: Normalization for all conv layers. See :func:`get_norm` for supported format.
        """
        super(BasicStem, self).__init__(in_channels, out_channels, 4)

        self.conv1 = layers.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=layers.get_norm(norm, out_channels),
        )
        weight_init.caffe2_msra_init(self.conv1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ResNet(layers.BaseModule):
    """`Deep Residual Learning for Image Recognition`_.

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385
    """
    # Mapping depth to block class and stage blocks
    settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (BottleneckBlock, (3, 4, 6, 3)),
        101: (BottleneckBlock, (3, 4, 23, 3)),
        152: (BottleneckBlock, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        stem: layers.BaseCNNBlock,
        stages: List[nn.Sequential],
        num_classes: Optional[int] = None,
        out_features: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            stem: A stem module.
            stages: Several (typically 4) stages, each contains a nn.Sequential of
                :class:`CNNBlockBase`.
            num_classes: If None, will not perform classification. Otherwise, will create a linear
                layer.
            out_features: Name of the layers whose outputs should be returned in forward. Can be
                anything in "stem", "linear", or "res2" ... If None, will return the output of the
                last layer.
        """
        super(ResNet, self).__init__()

        if len(stages) == 0:
            logger.warning('None residual stages provided. Maybe something wrong!')

        self._num_classes = num_classes

        output_shape = {}

        name = 'stem'
        self.add_module(name, stem)
        current_channels = stem.out_channels
        current_stride = stem.stride
        output_shape[name] = layers.ShapeSpec(channels=current_channels, stride=current_stride)

        self._stage_names = []
        for i, stage in enumerate(stages, start=2):
            if len(stage) == 0:
                raise ValueError('Stage is empty')
            for block in stage:
                if not isinstance(block, layers.BaseCNNBlock):
                    raise TypeError('Block should be CNNBlockBase. Got {}'.format(type(block)))

            name = 'res{}'.format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            current_channels = stage[-1].out_channels
            current_stride = current_stride * np.prod([b.stride for b in stage])
            output_shape[name] = layers.ShapeSpec(channels=current_channels, stride=current_stride)

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(current_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = 'linear'
            output_shape[name] = layers.ShapeSpec(channels=num_classes)

        if out_features is None:
            out_features = [name]
        assert len(out_features) != 0

        children = [x[0] for x in self.named_children()]
        for out_feature in out_features:
            assert out_feature in children, 'Available children: {}'.format(', '.format(children))
        self._output_shape = {k: v for k, v in output_shape.items() if k in out_features}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._output_shape:
            outputs['stem'] = x

        for name in self._stage_names:
            x = getattr(self, name)(x)
            if name in self._output_shape:
                outputs[name] = x

        if self._num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if 'linear' in self._output_shape:
                outputs['linear'] = x

        return outputs

    @classmethod
    def make_stage(
        cls,
        block_class: Type[Union[BasicBlock, BottleneckBlock]],
        num_blocks: int,
        first_stride: int,
        *,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ) -> nn.Sequential:
        """
        Args:
            block_class: A subclass of :class:`CNNBlockBase` that's used to create all blocks in
                this stage. A module of this type must not change spatial resolution of inputs
                unless its stride != 1.
            num_blocks: Number of blocks in this stage.
            first_stride: The stride of the first block. The other blocks will have stride=1.
            in_channels: Input channels of the entire stage.
            out_channels: Output channels of **every block** in the stage.
            **kwargs: Other arguments passed to the constructor of `block_class`.

        Returns:
            nn.Sequential: List of blocks.
        """
        if 'stride' in kwargs:
            raise ValueError('Stride of blocks in make_stage cannot be changed.')

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                block_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=first_stride if i == 0 else 1,
                    **kwargs,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def freeze(self, freeze_at: int = 0) -> 'ResNet':
        """Freezes the first several stages of the ResNet. Commonly used in fine-tuning.

        Layers that produce the same feature map spatial size are defined as one 'stage' by
        `Feature Pyramid Networks for Object Detection`_.

        Args:
            freeze_at: Number of stages to freeze. `1` means freezing the stem. `2` means freezing
                the stem and one residual stage, etc.

        Returns:
            This module itself

        _`Feature Pyramid Networks for Object Detection`:
            https://arxiv.org/abs/1612.03144
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, name in enumerate(self._stage_names, start=2):
            if freeze_at >= idx:
                stage = getattr(self, name)
                for block in stage.children():
                    block.freeze()
        return self

    @property
    def output_shape(self) -> Dict[str, layers.ShapeSpec]:
        return self._output_shape


@BackboneRegistry.register('ResNet_Backbone')
def build_resnet_backbone(
    depth: int = 50,
    in_channels: int = 3,
    stem_out_channels: int = 64,
    res2_out_channels: int = 256,
    norm: Union[str, Callable] = 'FrozenBN',
    num_groups: int = 1,
    width_per_group: int = 64,
    stride_in_1x1: bool = True,
    res5_dilation: int = 1,
    freeze_at: int = 2,
    out_features: List[str] = ('res4',),
    num_classes: Optional[int] = None,
) -> ResNet:
    """
    Args:
        depth: Depth of ResNet layers, can be 18, 34, 50, 101, or 152.
        in_channels: Number of input channels of ResNet.
        stem_out_channels: Number of output channels of stem. For R18 and R34, this is needs to be
            set to 64.
        res2_out_channels: Number of output channels of res2.
        norm: Normalization for all conv layers. See :func:`get_norm` for supported format.
        num_groups: Number of groups, 1 -> ResNet, 2 -> ResNeXt.
        width_per_group: Baseline width of each group.
        stride_in_1x1: Place the stride 2 conv on the first 1x1 filter. Use True only for the
            original MSRA ResNet; use False for C2 and Torch models.
        res5_dilation: Apply dilation in stage "res5".
        freeze_at: Freeze the first several stages so they are not trained. There are 5 stages in
            ResNet. The first is a convolution, and the following stages are each group of residual
            blocks.
        num_classes: See :class:`ResNet`.
        out_features: See :class:`ResNet`.
    """
    if res5_dilation not in (1, 2):
        raise ValueError('res5_dilation can only be 1 or 2. Got {}'.format(res5_dilation))
    if depth in (18, 34):
        if res2_out_channels != 64:
            raise ValueError('Must set res2_out_channels = 64 for R18/R34')
        if res5_dilation != 1:
            raise ValueError('Must set res5_dilation = 1 for R18/R34')
        if num_groups != 1:
            raise ValueError('Must set num_groups = 1 for R18/R34')

    block_class, stage_blocks = ResNet.settings[depth]

    stem = BasicStem(in_channels, stem_out_channels, norm)

    in_channels = stem_out_channels
    out_channels = res2_out_channels
    bottleneck_channels = num_groups * width_per_group

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause all-reduce to fail
    feature_to_idx = {'res2': 2, 'res3': 3, 'res4': 4, 'res5': 5}
    out_stage_idx = [feature_to_idx.get(f, -1) for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if stage_idx == 2 or (stage_idx == 5 and dilation == 2) else 2
        stage_kwargs = {
            'block_class': block_class,
            'num_blocks': stage_blocks[idx],
            'first_stride': first_stride,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'norm': norm,
        }
        # BottleneckBlock is used by R50/R101/R152
        if depth in (50, 101, 152):
            stage_kwargs['bottleneck_channels'] = bottleneck_channels
            stage_kwargs['stride_in_1x1'] = stride_in_1x1
            stage_kwargs['dilation'] = dilation
            stage_kwargs['num_groups'] = num_groups

        stage = ResNet.make_stage(**stage_kwargs)
        stages.append(stage)
        # Update arguments for next stage
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

    return ResNet(stem, stages, num_classes, out_features).freeze(freeze_at)
