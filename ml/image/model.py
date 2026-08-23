import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn
class ImageDeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT); self.backbone.classifier[1]=nn.Linear(self.backbone.classifier[1].in_features,2)
    def forward(self,x): return self.backbone(x)
