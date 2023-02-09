import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class NVidiaNet(nn.Module):
  def __init__(self):
    super(NVidiaNet, self).__init__()
    self.encoder = nn.Sequential(
      nn.BatchNorm2d(3),
      nn.Conv2d(3, 24, (5, 5), stride=(2, 2)),
      nn.ELU(),
      nn.Conv2d(24, 36, (5, 5), stride=(2, 2)),
      nn.ELU(),
      nn.Conv2d(36, 48, (5, 5), stride=(2, 2)),
      nn.ELU(),
      nn.Dropout(0.4),
      nn.Conv2d(48, 64, (3, 3), stride=(1, 1), padding=1),
      nn.ELU(),
      nn.Conv2d(64, 64, (3, 3), stride=(1, 1)),
      nn.ELU(),
    )
    self.decoder = nn.Sequential(
      nn.Linear(64*15, 50),
      nn.ELU(),
      nn.Linear(50, 10),
      nn.ELU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = torch.flatten(x, start_dim=1)
    x = self.decoder(x)
    return x

class SpeedNet(nn.Module):
  def __init__(self):
    super(SpeedNet, self).__init__()
    resnet = list(resnet34(weights=ResNet34_Weights.DEFAULT).children())[:-4] #except last 2 layers    
    self.conv_seq1 = nn.Sequential(
      nn.BatchNorm2d(3),
      nn.Conv2d(3, 3, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False),
      nn.BatchNorm2d(3),
      nn.ELU())
    self.resnet = nn.Sequential(*resnet)
    self.conv_seq2 = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ELU(),
      nn.Conv2d(64, 64, 1, stride=2, bias=False),
      nn.BatchNorm2d(64),
      nn.ELU())
    self.linear = nn.Sequential(
      nn.Linear(64*3, 64),
      nn.BatchNorm1d(64),
      nn.Dropout(0.5),
      nn.ELU(),
      nn.Linear(64, 32),
      nn.BatchNorm1d(32),
      nn.Dropout(0.5),
      nn.ELU(),
      nn.Linear(32, 1))

  def forward(self, x):
    x = self.conv_seq1(x)
    x = self.resnet(x)
    x = self.conv_seq2(x)
    x = torch.flatten(x, start_dim=1)
    x = self.linear(x)
    return x

if __name__ == "__main__":
  NVidiaNet()(torch.zeros((64, 3, 50, 160)))
  exit()