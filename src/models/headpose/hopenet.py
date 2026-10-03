"""
Head-Pose Estimation Models for Continuous Yaw Regression
Implements:
1. HopeNet-Yaw: ResNet50 backbone with binned classification (66 bins over [-99°, +99°])
   and continuous softmax expectation (Ruiz et al., CVPRW 2018).
2. ResNet18-Yaw: Lightweight ResNet18 direct continuous yaw regressor baseline.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    resnet18,
    ResNet50_Weights,
    ResNet18_Weights,
)


class HopeNetYaw(nn.Module):
    """
    HopeNet architecture specialized for Yaw orientation estimation.
    Uses ResNet50 backbone with a 66-class classification head over [-99°, +99°] (3° bin width).
    Continuous yaw is computed as the expected value over the softmax distribution:
        theta_pred = sum(P_i * bin_center_i)
    """

    def __init__(self, num_bins: int = 66, min_angle: float = -99.0, max_angle: float = 99.0, pretrained: bool = True):
        super().__init__()
        self.num_bins = num_bins
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.bin_width = (max_angle - min_angle) / num_bins

        # Precompute bin centers: [-97.5, -94.5, ..., +97.5]
        bin_centers = torch.tensor(
            [min_angle + (i + 0.5) * self.bin_width for i in range(num_bins)],
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", bin_centers)

        # ResNet-50 backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Keep everything up to avgpool
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.fc_yaw = nn.Linear(2048, num_bins)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, continuous_yaw_degrees)
        """
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

        logits = self.fc_yaw(x)
        probs = F.softmax(logits, dim=1)
        expected_yaw = torch.sum(probs * self.bin_centers, dim=1)

        return logits, expected_yaw

    def angle_to_bin(self, angles: torch.Tensor, strict: bool = True) -> torch.Tensor:
        """
        Converts continuous angles to discrete bin indices over half-open interval [min_angle, max_angle).
        Valid angles: min_angle <= angle < max_angle (e.g. [-99.0, +99.0)).
        If strict=True: strictly validates that all angles are within [min_angle, max_angle).
        angle == +99.0 produces bin 66 (out-of-range for 66 bins 0..65) and MUST raise ValueError.
        Silent clipping and silent mapping of +99.0 are strictly prohibited.
        """
        if torch.isnan(angles).any() or torch.isinf(angles).any():
            raise ValueError("Input angles contain NaN or Inf")
        if strict:
            out_of_range_mask = (angles < self.min_angle) | (angles >= self.max_angle)
            if out_of_range_mask.any():
                violating = angles[out_of_range_mask]
                raise ValueError(
                    f"HopeNet range violation: {len(violating)} angle(s) fell outside "
                    f"[{self.min_angle}, {self.max_angle}) (e.g. {violating[0].item():.4f}°). "
                    f"Silent clipping is prohibited under V4C safety rules."
                )
        bins = torch.floor((angles - self.min_angle) / self.bin_width).long()
        return bins


class ResNet18Yaw(nn.Module):
    """
    Lightweight ResNet18 continuous circular yaw regressor baseline.
    Outputs unit-circle vector (sin_yaw, cos_yaw), decoded via atan2 to canonical [-180, 180) degrees.
    Eliminates +/-180 degree boundary discontinuities.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # 2 outputs: [sin(yaw), cos(yaw)]
        self.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (raw_vector, canonical_yaw_degrees)
        - raw_vector: (B, 2) [sin_hat, cos_hat]
        - canonical_yaw_degrees: (B,) Euler angle in [-180, 180) degrees
        """
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
        raw_vec = self.fc(x)  # (B, 2) -> (sin, cos)

        # Normalize predicted vector to unit circle with explicit epsilon protection
        norm = torch.norm(raw_vec, p=2, dim=1, keepdim=True)
        unit_vec = raw_vec / (norm + 1e-7)
        sin_val = unit_vec[:, 0]
        cos_val = unit_vec[:, 1]

        # Decode via atan2: (-pi, pi]
        yaw_rad = torch.atan2(sin_val, cos_val)
        yaw_deg = yaw_rad * (180.0 / torch.pi)

        # Canonicalize to [-180, 180)
        canonical_yaw = torch.remainder(yaw_deg + 180.0, 360.0) - 180.0

        return raw_vec, canonical_yaw


