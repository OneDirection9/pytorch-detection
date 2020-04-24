# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from foundation.backends.torch.utils import weight_init
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BackboneStash
from ..utils import get_norm

__all__ = ['ResNet']


def conv3x3(in_planes, out_planes, stride=1, dilation=1, groups=1):
    """3x3 convolution with padding.

    The output size is computed as follows (floor division):
        out = (in + 2 * padding - (kernel_size - 1) * dilation - 1) / stride + 1
            = (in + 2 * dilation - (3 - 1) * dilation - 1) / stride + 1
            = (in - 1) / stride + 1
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic block used by res-18 and res-34 typically.

    Some not used arguments are added to keep the arguments the same as Bottleneck, i.e.
    bottleneck_channels, groups, is_msra. Keep the default settings, otherwise it will
    raise a ValueError.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bottleneck_channels (int): Number of bottleneck channels. Not used.
        stride (int, optional): Number of stride. Default: 1
        dilation (int, optional): Number of dilation. Default: 1
        groups (int, optional): Number of groups. Not used. Default: 1
        norm (str, optional): Name of normalization. Default: `BN`
        is_msra (bool, optional): Apply stride in first conv layer or second conv layer.
            Not used. Default: ``True``
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        dilation=1,
        groups=1,
        norm='BN',
        is_msra=True
    ):
        super(BasicBlock, self).__init__()

        if out_channels != bottleneck_channels:
            raise ValueError('BasicBlock only supports out_channels == bottleneck_channels')
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        if not is_msra:
            raise ValueError('BasicBlock only supports is_msra=True')

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                get_norm(norm, out_channels),
            )
        else:
            self.downsample = None

        self.conv1 = conv3x3(in_channels, out_channels, stride, dilation=dilation)
        self.norm1 = get_norm(norm, out_channels)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = get_norm(norm, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck used by res-50, res-101, res-152 typically.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bottleneck_channels (int): Number of bottleneck channels.
        stride (int, optional): Number of stride. Default: 1
        dilation (int, optional): Number of dilation. Default: 1
        groups (int, optional): Number of groups. Default: 1
        norm (str, optional): Name of normalization. Default: `BN`
        is_msra (bool, optional): Apply stride in first conv layer for original MSRA
            ResNet or second conv layer for C2 and Torch Models. Default: ``True``
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        dilation=1,
        groups=1,
        norm='BN',
        is_msra=True
    ):
        super(Bottleneck, self).__init__()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                get_norm(norm, out_channels),
            )
        else:
            self.downsample = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        if is_msra:
            conv1_stride, conv2_stride = stride, 1
        else:
            conv1_stride, conv2_stride = 1, stride

        self.conv1 = conv1x1(in_channels, bottleneck_channels, conv1_stride)
        self.norm1 = get_norm(norm, bottleneck_channels)

        self.conv2 = conv3x3(
            bottleneck_channels,
            bottleneck_channels,
            conv2_stride,
            dilation=dilation,
            groups=groups
        )
        self.norm2 = get_norm(norm, bottleneck_channels)

        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.norm3 = get_norm(norm, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


@BackboneStash.register('ResNet')
class ResNet(nn.Module):
    """`Deep Residual Learning for Image Recognition`_.

    Args:
        depth (int): Depth of ResNet, typically is 18, 34, 50, 101, 152.
        in_channels (int, optional): Number of input image channels. Default: 3
        stem_out_channels (int, optional): Number of conv output channels in stem which
            consists of Conv -> Norm -> ReLU -> MaxPool layers. It is also the input
            channels of residual layer. Default: 64
        res2_out_channels (int, optional): Number of output channels in first stage, i.e.
            'res2' layer. This will be doubled by stage. Default: 256
        num_classes (int, optional): If None, will not perform classification.
        out_features (list[str], optional): Name of the layers whose outputs should be
            returned in forward. Can be anything in 'stem', 'res2', 'res3', 'res4',
            'res5'. If `num_classes` is provided, another 'linear` output is appended.
            Default: ('res2', 'res3', 'res4', 'res5')
        freeze_stages (int, optional):  The 1-indexed stages that less and equal to this
            will be frozen. If <= 0, no stages will be frozen. Default: 0
        groups (int, optional): Number of groups. Default: 1
        base_width_per_group (int, optional): Basic width of each group. This will be
            doubled by stage. Default: 64
        norm (str, optional): Name of normalization. Default: `BN`
        norm_eval (bool, optional): If ``True``, normalization layer is set to eval mode.
            Default: ``True``
        is_msra (bool, optional): Apply stride in first conv layer for original MSRA
            ResNet or second conv layer for C2 and Torch Models. Default: ``True``

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385
    """
    settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=64,
        res2_out_channels=256,
        num_classes=None,
        out_features=('res2', 'res3', 'res4', 'res5'),
        freeze_stages=0,
        groups=1,
        base_width_per_group=64,
        norm='BN',
        norm_eval=True,
        is_msra=True,
        zero_init_residual=True
    ):
        super(ResNet, self).__init__()

        self._depth = depth
        self._res2_out_channels = res2_out_channels
        self._num_classes = num_classes
        self._freeze_stages = freeze_stages
        self._groups = groups
        self._norm = norm
        self._norm_eval = norm_eval
        self._is_msra = is_msra
        self._zero_init_residual = zero_init_residual

        # Make stem which has stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out_channels, 7, stride=2, padding=3, bias=False),
            get_norm(norm, stem_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        block_class, stage_blocks = self.settings[depth]
        if num_classes is not None:  # Need all stages to do classification
            max_stage = 5
        else:
            name_to_idx = {'stem': 1, 'res2': 2, 'res3': 3, 'res4': 4, 'res5': 5}
            out_stages_idx = [name_to_idx[name] for name in out_features]
            max_stage = max(out_stages_idx)

        in_channels = stem_out_channels
        out_channels = res2_out_channels
        bottleneck_channels = groups * base_width_per_group

        # The residual layer in format: res<stage>, where stage has stride = 2 ** stage
        self._res_layer_names = []
        for idx, stage_idx in enumerate(range(2, max_stage + 1)):
            num_blocks = stage_blocks[idx]
            stride = 1 if idx == 0 else 2
            dilation = 1
            stage = self.make_stage(
                block_class,
                num_blocks,
                in_channels,
                out_channels,
                bottleneck_channels,
                stride=stride,
                dilation=dilation
            )
            name = 'res{}'.format(stage_idx)
            self.add_module(name, stage)
            self._res_layer_names.append(name)

            in_channels = out_channels
            out_channels *= 2
            bottleneck_channels *= 2

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(out_channels, num_classes)

            out_features += ('linear',)

        self._out_features = out_features
        children_names = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            if out_feature not in children_names:
                raise ValueError('Available children: {}'.format(children_names))

        self.reset_parameters()
        self.freeze()

    def make_stage(
        self,
        block_class,
        num_blocks,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        dilation=1
    ):
        blocks = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            blocks.append(
                block_class(
                    in_channels,
                    out_channels,
                    bottleneck_channels,
                    stride=stride,
                    dilation=dilation,
                    groups=self._groups,
                    norm=self._norm,
                    is_msra=self._is_msra,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_normal_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                weight_init.constant_init(m, 1)

        if self._num_classes is not None:
            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)

        if self._zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    weight_init.constant_init(m.norm2, 0)
                elif isinstance(m, Bottleneck):
                    weight_init.constant_init(m.norm3, 0)

    def freeze(self):
        if self._freeze_stages >= 1:
            self.stem.eval()
            for p in self.stem.parameters():
                p.requires_grad = False

        if self._freeze_stages >= 2:
            for stage_idx in range(2, self._freeze_stages + 1):
                m = getattr(self, 'res{}'.format(stage_idx))
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (Tensor[N, C, H, W]): Input tensor.

        Returns:
            tuple[Tensor]: Tuple of feature map tensor in high to low resolution order.
        """
        outputs = []
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs.append(x)

        for name in self._res_layer_names:
            stage = getattr(self, name)
            x = stage(x)
            if name in self._out_features:
                outputs.append(x)

        if self._num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            outputs.append(x)

        return tuple(outputs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self.freeze()

        if mode and self._norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
