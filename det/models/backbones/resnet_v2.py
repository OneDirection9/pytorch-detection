from __future__ import absolute_import, division, print_function

from typing import Callable, Dict, List, Optional, Union

import torch
from foundation.nn import weight_init
from torch.nn import functional as F

from det import layers
from det.layers import ShapeSpec


class BasicBlock(layers.CNNBlockBase):
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


class BottleneckBlock(layers.CNNBlockBase):
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
            bottleneck_channels: Number of output channels for the 3x3 'bottleneck' conv layers.
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
            shortcut = None

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(layers.CNNBlockBase):
    """The standard ResNet stem (layers before the first residual block)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        norm: Union[str, Callable] = 'BN'
    ) -> None:
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


class ResNet(layers.Module):
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
        depth: int = 50,
        in_channels: int = 3,
        stem_out_channels: int = 64,
        res2_out_channels: int = 256,
        num_classes: Optional[int] = None,
        out_features: List[str] = ('res4',),
        norm: Union[str, Callable] = 'FrozenBN',
        num_groups: int = 1,
        width_per_group: int = 64,
        stride_in_1x1: bool = True,
        res5_dilation: int = 1,
        freeze_at: int = 2,
    ) -> None:
        """
        Args:
            depth:
            in_channels:
            stem_out_channels:
            res2_out_channels:
            num_classes:
            out_features:
            norm:
            num_groups:
            width_per_group:
            stride_in_1x1:
            res5_dilation:
            freeze_at:
        """
        super(ResNet, self).__init__()

        if res5_dilation not in (1, 2):
            raise ValueError('res5_dilation can only be 1 or 2. Got {}'.format(res5_dilation))
        if depth in [18, 34]:
            if res2_out_channels != 64:
                raise ValueError('Must set res2_out_channels = 64 for R18/R34')
            if res5_dilation != 1:
                raise ValueError('Must set res5_dilation = 1 for R18/R34')
            if num_groups != 1:
                raise ValueError('Must set num_groups = 1 for R18/R34')

        block_class, stage_blocks = self.settings[depth]

        self.stem = BasicStem(in_channels, stem_out_channels, norm)

        in_channels = stem_out_channels
        out_channels = res2_out_channels
        bottleneck_channels = num_groups * width_per_group

        # Avoid creating variables without gradients
        # It consumes extra memory and may cause all-reduce to fail
        feature_to_idx = {'res2': 2, 'res3': 3, 'res4': 4, 'res5': 5}
        out_stage_idx = [feature_to_idx[f] for f in out_features]
        max_stage_idx = max(out_stage_idx)
        stages = []
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

            if depth in (50, 101, 152):
                stage_kwargs['bottleneck_channels'] = bottleneck_channels
                stage_kwargs['stride_in_1x1'] = stride_in_1x1
                stage_kwargs['dilation'] = dilation
                stage_kwargs['num_groups'] = num_groups

            blocks = self.make_stage(**stage_kwargs)
            in_channels = out_channels
            out_channels *= 2
            bottleneck_channels *= 2
            stages.append(blocks)

    @classmethod
    def make_stage(
        cls, block_class, num_blocks, first_stride, *, in_channels, out_channels, **kwargs
    ):
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
        return blocks

    @property
    def output_shape(self) -> Dict[str, ShapeSpec]:
        return dict()
