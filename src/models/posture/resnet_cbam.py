"""
ResNet with CBAM (Convolutional Block Attention Module)
Provides ResNet-18 + CBAM and ResNet-50 + CBAM architectures with full
pre-trained weight compatibility from torchvision.
"""

from typing import Type, Union, List, Optional
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)

from src.models.posture.cbam import CBAM


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockCBAM(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        cbam_ratio: int = 16,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlockCBAM only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlockCBAM")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes * self.expansion, ratio=cbam_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.cbam(out)

        return out


class BottleneckCBAM(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        cbam_ratio: int = 16,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes * self.expansion, ratio=cbam_ratio)
        if hasattr(self.cbam.ca, "fc2") and hasattr(self.cbam.ca.fc2, "weight"):
            nn.init.zeros_(self.cbam.ca.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCBAM(nn.Module):
    """
    Standard ResNet backbone with CBAM attention integrated into each residual block.
    Retains torchvision-compatible state_dict naming for seamless transfer learning.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlockCBAM, BottleneckCBAM]],
        layers: List[int],
        num_classes: int = 4,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckCBAM):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockCBAM):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlockCBAM, BottleneckCBAM]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cbam(num_classes: int = 4, pretrained: bool = True) -> ResNetCBAM:
    """Constructs ResNet-18 with CBAM attention blocks."""
    model = ResNetCBAM(BasicBlockCBAM, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        # Exclude fc layer weights due to class count difference
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        res = model.load_state_dict(filtered_state_dict, strict=False)
        # Verify that all backbone conv and bn layers matched successfully
        assert len(res.unexpected_keys) == 0, f"Unexpected keys when loading ResNet18: {res.unexpected_keys}"
    return model


def resnet50_cbam(num_classes: int = 4, pretrained: bool = True) -> ResNetCBAM:
    """Constructs ResNet-50 with CBAM attention blocks."""
    model = ResNetCBAM(BottleneckCBAM, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        res = model.load_state_dict(filtered_state_dict, strict=False)
        assert len(res.unexpected_keys) == 0, f"Unexpected keys when loading ResNet50: {res.unexpected_keys}"

    # Explicitly ensure all CBAM ChannelAttention fc2 projection weights are zero-initialized
    # to guarantee identity residual mapping and prevent mixed-precision gradient explosion.
    for m in model.modules():
        if hasattr(m, "fc2") and isinstance(m.fc2, nn.Conv2d):
            nn.init.zeros_(m.fc2.weight)

    return model
