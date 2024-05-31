"""
Contains PyTorch model code to instantiate a Alexnet model.
"""
import torch
from torch import nn

class Alexnet(nn.Module):
  def __init__(self, num_classes: int) -> None:
      super().__init__()
      self.features = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace = True),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

          nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
          nn.ReLU(inplace = True),
          nn.MaxPool2d(kernel_size=3,stride=2, padding=0),

          nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace = True),

          nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace = True),

          nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3,stride=2, padding=0)
      )

      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=9216,out_features=4096),
          nn.ReLU(inplace=True),
          nn.Linear(in_features=4096,out_features=4096),
          nn.ReLU(inplace=True),
          nn.Linear(in_features=4096,out_features=num_classes)
      )

  def forward(self, x: torch.Tensor):
      x = self.classifier(self.features(x))
      return x
