#!/usr/bin/env python3
from preprocess import OUTPUT_PATH, DATA_PATH
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

# class Sigmoid(nn.Module):
#     def __init__(self, multiplier):
#       super(Sigmoid, self).__init__()
#       self.sigmoid = nn.Sigmoid()  
#       self.multiplier = multiplier

#     def forward(self, x):
#       return self.sigmoid(x) * self.multiplier

# class ResNet(torch.nn.Module):
#   def __init__(self, name='resnet', *args, **kwargs) -> None:
#     super().__init__(*args, **kwargs)
#     self.name = name
#     self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
#     in_features = self.resnet.fc.in_features
#     self.resnet.fc = nn.Identity()
#     self.fc = nn.Sequential(
#         nn.Linear(in_features, 256),   
#         Sigmoid(40),
#         nn.Dropout(0.3),
#         nn.Linear(256, 1)   
#     )
    
#   def forward(self, x):
#     x = x.permute(0, 3, 1, 2)
#     x = self.resnet(x)
#     x = self.fc(x)
#     return x

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict


def build_model(pretrained=True, fine_tune=True, num_classes=1):
  if pretrained:
    print('INFO: Loading pre-trained weights')
  else:
    print('INFO: Not Loading pre-trained weights')
  
  model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
  for params in model.parameters():
    if fine_tune:
      params.requires_grad = True
    else:
      params.requires_grad = False
  
  model.classifier[1] = nn.Linear(in_features=1280, out_features=1)
  return model


def load_filename(filename, i):
  print('loading data', filename)
  x = torch.load(f'{OUTPUT_PATH}/{filename}')["x"]
  y = torch.Tensor(np.genfromtxt(os.path.join(DATA_PATH, filename.replace('.pt', '.txt'))))
  return x, y

def load_speed_data():
  data = []
  for i, filename in enumerate(os.listdir(OUTPUT_PATH)):
    if filename.endswith('pt'):
      x, y = load_filename(filename, i+1)
      data.append((x, y))
  return data

def get_minibatch(set, bs=64):
  xs, ys = [], []
  for _ in range(bs):
    src, val = random.choice(set)
    sel = random.randint(0, src.shape[0]-1)
    xs.append(src[sel:sel+1])
    ys.append(val[sel:sel+1])
  return torch.Tensor(np.concatenate(xs, axis=0)), torch.Tensor(np.array(ys))

if __name__ == '__main__':
  train_set = load_speed_data()
  # tx, ty = get_minibatch(train_set)
  # print(tx.shape, ty.shape, tx.dtype, ty.dtype)

  model = build_model().to('mps')
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

  for i in (t:=tqdm(range(600))):
    optimizer.zero_grad()
    tx, ty = get_minibatch(train_set)
    tx, ty = tx.to('mps'), ty.to('mps')
    out = model(tx.permute(0, 3, 1, 2))
    loss = F.mse_loss(out, ty)
    loss.backward()
    optimizer.step()
    t.set_description(f'train loss {loss.item():.2f}')
