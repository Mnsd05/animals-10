# Implement mobilenet using pytorch
"""
Contains PyTorch model code to instantiate a Mobilenetv1 model.
"""
import torch
from torch import nn

class SeparableDepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SeparableDepthWiseConv(32, 64, 3, 1, 1),
            SeparableDepthWiseConv(64, 128, 3, 2, 1),
            SeparableDepthWiseConv(128, 128, 3, 1, 1),
            SeparableDepthWiseConv(128, 256, 3, 2, 1),
            SeparableDepthWiseConv(256, 256, 3, 1, 1),
            SeparableDepthWiseConv(256, 512, 3, 2, 1),
            SeparableDepthWiseConv(512, 512, 3, 1, 1),
            SeparableDepthWiseConv(512, 512, 3, 1, 1),
            SeparableDepthWiseConv(512, 512, 3, 1, 1),
            SeparableDepthWiseConv(512, 512, 3, 1, 1),
            SeparableDepthWiseConv(512, 1024, 3, 2, 1),
            SeparableDepthWiseConv(1024, 1024, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

        self.name = "MobileNetV1"
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
        
