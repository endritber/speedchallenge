#!/usr/bin/env python3
from preprocess import OUTPUT_PATH, DATA_PATH
from tqdm import tqdm
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

def build_model(pretrained=True, fine_tune=True, num_classes=1):
  if pretrained:
    print('INFO: loading pre-trained weights')
  
  model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
  for params in model.parameters():
    if fine_tune:
      params.requires_grad = True
    else:
      params.requires_grad = False
  
  model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)
  print(model._get_name())
  return model

def run(data, kfold=False):
  if not kfold:
    n = len(data[0])
    split = int(n*0.9)
    train_data = (data[0][0:split], data[1][0:split])
    val_data = (data[0][split:n], data[1][split:n])
    train(train_data, val_data)

def test_step(tx, ty):
  with torch.no_grad():
    model.eval()
    out = model(tx)
    loss = F.mse_loss(out, ty)
    return loss.item()

def train_step(tx, ty):
  optimizer.zero_grad()
  out = model(tx)
  loss = F.mse_loss(out, ty)
  loss.backward()
  optimizer.step()
  return loss.item()

def train(train_data, val_data):
  epochs = 1600
  batch_size = 32

  tx, ty = get_minibatch(val_data, 32)
  last = time.monotonic()
  tlosses, vlosses, i = [], [], 1
  for epoch in (t:=tqdm(range(1, epochs+1))):
    val_loss = test_step(tx, ty)
    x, y = get_minibatch(train_data, batch_size)
    train_loss = train_step(x, y)
    tlosses.append(float(train_loss))
    vlosses.append(float(val_loss))
    t.set_description(f'Training -> epoch: [{epoch}|{epochs}] | train_loss: {train_loss:.3f} | val_loss: {val_loss:.3f}')
    epoch_time = time.monotonic() - last     
    if i % 50 == 0:
      print(f"Training this model takes {epoch_time * epochs // 60} min, done in {epoch_time * (epochs - epoch) // 60} min")
    last = time.monotonic()
    i += 1

  print('Saving model...')
  torch.save(model.state_dict(), 'demo/speed_efficientnet.pt')
  import matplotlib.pyplot as plt
  plt.plot(vlosses)
  plt.plot(tlosses)
  plt.show()

def load_filename(filename, i):
  print('loading data', filename)
  x = torch.load(f'{OUTPUT_PATH}/{filename}')["x"]
  y = torch.Tensor(np.genfromtxt(os.path.join(DATA_PATH, filename.replace('.pt', '.txt'))))
  return x, y

def load_speed_data():
  xs, ys = [], []
  for i, filename in enumerate(os.listdir(OUTPUT_PATH)):
    if filename.endswith('pt'):
      x, y = load_filename(filename, i+1)
      xs.append(x); ys.append(y)
  return (xs, ys)

def get_minibatch(set, bs=64, device='mps'):
  set = [(x, y) for x, y in zip(set[0], set[1])]
  xs, ys = [], []
  for _ in range(bs):
    src, val = random.choice(set)
    sel = random.randint(0, src.shape[0]-1)
    xs.append(src[sel:sel+1])
    ys.append(val[sel:sel+1])
  return torch.Tensor(np.concatenate(xs, axis=0)).permute(0, 3, 1, 2).to(device), torch.Tensor(np.array(ys)).to(device)

if __name__ == '__main__':
  train_set = load_speed_data()
  lr = 1e-4
  model = build_model().to('mps')
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  run(train_set)
  