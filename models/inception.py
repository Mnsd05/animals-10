
# Implement inceptionet using pytorch
import torch
from torch import nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, reduce_3x3, out_5x5, reduce_5x5, out_pool):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, 1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, 1),
            nn.BatchNorm2d(reduce_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, 3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, 1),
            nn.BatchNorm2d(reduce_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, 5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, 1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)
    
class InceptionNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            InceptionModule(192, 64, 128, 96, 32, 16, 32),
            InceptionModule(256, 128, 192, 128, 96, 32, 64),
            nn.MaxPool2d(3, 2, 1),
            InceptionModule(480, 192, 208, 96, 48, 16, 64),
            InceptionModule(512, 160, 224, 112, 64, 24, 64),
            InceptionModule(512, 128, 256, 128, 64, 24, 64),
            InceptionModule(512, 112, 288, 144, 64, 32, 64),
            InceptionModule(528, 256, 320, 160, 128, 32, 128),
            nn.MaxPool2d(3, 2, 1),
            InceptionModule(832, 256, 320, 160, 128, 32, 128),
            InceptionModule(832, 384, 384, 192, 128, 48, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
        self.name = "InceptionNet"
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
