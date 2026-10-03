"""
MobileNet Posture Classifier Baseline
Uses torchvision's official MobileNetV3-Small architecture with official
ImageNet pre-trained weights as the lightweight edge posture classifier baseline.
(Serves as the well-supported lightweight edge candidate required by V4C Section 6).
"""

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


def mobilenet_posture(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Constructs MobileNetV3-Small classifier adapted to the 4-class posture ontology.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    
    # Replace the final linear layer in the classifier
    # mobilenet_v3_small classifier is:
    # (0): Linear(in_features=576, out_features=1024, bias=True)
    # (1): Hardswish()
    # (2): Dropout(p=0.2, inplace=True)
    # (3): Linear(in_features=1024, out_features=1000, bias=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model
