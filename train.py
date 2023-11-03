#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from preprocess import OUTPUT_PATH, DATA_PATH
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

LOAD = os.getenv('LOAD') != None
TIMESTAMP = int(time.time())

# https://github.com/pytorch/vision/issues/7744
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

def build_model(pretrained=True, fine_tune=True, num_classes=1):
  if pretrained: print('INFO: loading pretrained model')
  model = torchvision.models.efficientnet_b0(pretrained=pretrained); print('Running:', model._get_name())
  for params in model.parameters():
    if fine_tune: params.requires_grad = True
    else: params.requires_grad = False
  model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)
  return model

# Add K-Fold strategy
def run(data, kfold=False):
  if not kfold:
    n = len(data[0])
    split = int(n*0.9)
    train_data = [torch.cat(data[0][0:split], dim=0), torch.cat(data[1][0:split], dim=0)]
    val_data = [torch.cat(data[0][split:n], dim=0), torch.cat(data[1][split:n], dim=0)]
    train(train_data, val_data)

def test_step(batch, data):
  losses = []
  with torch.no_grad():
    model.eval()
    for samples in (t:=tqdm(batch)):
      inputs, targets = load_batch_samples(samples, data, device)
      outputs = model(inputs).squeeze(1)
      loss = F.mse_loss(outputs, targets)
      losses.append(loss.item())
      t.set_description(f'Validating: mse_loss {loss.item():.3f}')
    return losses

def train_step(batch, data, epoch):
  model.train()
  losses = []
  for samples in (t:=tqdm(batch)):
    optimizer.zero_grad()
    inputs, targets = load_batch_samples(samples, data, device)
    outputs = model(inputs).squeeze(1)
    loss = F.mse_loss(outputs, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    t.set_description(f'Training: [{epoch}|{epochs}] | mse_loss: {loss.item():.3f}')      
  return losses

def train(train_data, val_data):
  tlosses, vlosses = [], []
  val_batches = load_batch_indexes(np.arange(val_data[0].shape[0]), batch_size=batch_size)
  last = time.monotonic()
  for epoch in range(1, epochs+1):
    val_losses = test_step(val_batches, val_data)
    train_losses = train_step(load_batch_indexes(np.arange(train_data[0].shape[0]), \
                batch_size=batch_size, shuffle=True), train_data, epoch)
    tlosses.extend(val_losses)
    vlosses.extend(train_losses)
    epoch_time = time.monotonic() - last     
    print(f"Training (time) in {epoch_time * epochs // 60} min | Done (time) in {epoch_time * (epochs - epoch) // 60} min"); last = time.monotonic()
    save_model(fn=f"models/{model._get_name()}_{TIMESTAMP}_{epoch}_{torch.mean(torch.tensor(val_losses)).item():.2f}.pt")
  
  import matplotlib.pyplot as plt
  plt.plot(vlosses)
  plt.plot(tlosses)
  plt.show()

def save_model(fn):
  torch.save(model.state_dict(), fn)
  print(f"saved model {fn} with size {os.path.getsize(fn)}")

def load_filename(filename):
  x = torch.load(f'{OUTPUT_PATH}/{filename}')["x"]
  y = torch.Tensor(np.genfromtxt(os.path.join(DATA_PATH, filename.replace('.pt', '.txt'))))
  return x, y

def load_data():
  xs, ys = [], []
  for filename in (t:=tqdm(os.listdir(OUTPUT_PATH))):
    t.set_description(f'loading {filename}')
    if filename.endswith('pt'):
      x, y = load_filename(filename)
      xs.append(x); ys.append(y)
  return (xs, ys)

def load_batch_indexes(indexes, batch_size, shuffle=False):
  if shuffle:
    np.random.shuffle(indexes)
  return np.array(indexes)[:len(indexes) // batch_size*batch_size].reshape(-1, batch_size)

def load_batch_samples(samples, data, device):
  X = data[0][samples].permute(0, 3, 1, 2).to(device)
  Y = data[1][samples].to(device)
  return X, Y

if __name__ == '__main__':
  device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
 
  epochs = 50
  batch_size = 32
  lr = 1e-4

  model = build_model(pretrained=True, fine_tune=True).to(device)
  if LOAD:
    model_path = sys.argv[-1]
    print('INFO: loading saved model', model_path)
    model.load_state_dict(torch.load(model_path))
 
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  dataset = load_data()
  run(dataset)
  