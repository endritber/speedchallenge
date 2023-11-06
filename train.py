#!/usr/bin/env python3
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from preprocess import OUTPUT_PATH, DATA_PATH
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from video import running_mean
from sklearn.metrics import mean_squared_error

WINDOWS=15
LOAD = os.getenv('LOAD') != None
TIMESTAMP = int(time.time())

# https://github.com/pytorch/vision/issues/7744
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

def build_model(fine_tune=True, weights=None):
  if weights: print('INFO: loading pretrained model')
  model = torchvision.models.efficientnet_b2(weights=weights); print('Running:', model._get_name())
  for params in model.parameters():
    if fine_tune: params.requires_grad = True
    else: params.requires_grad = False
  model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)
  return model

# TODO: add K-Fold strategy
def run(data, kfold=False):
  n = len(data[0])
  split = int(n*0.9)
  train_data = [torch.cat(data[0][0:split], dim=0), torch.cat(data[1][0:split], dim=0)]
  val_data = [torch.cat(data[0][split:n], dim=0), torch.cat(data[1][split:n], dim=0)]
  train(train_data, val_data)

def rolling_mse_loss(outputs, targets):
  outputs = outputs.detach().cpu().numpy()
  targets = targets.detach().cpu().numpy()
  rolling_outputs = running_mean(outputs, WINDOWS)
  rolling_loss = mean_squared_error(rolling_outputs, targets)
  return rolling_loss

def validation_step(batch, data):
  model.eval()
  losses, rlosses = [], []
  with torch.no_grad():
    for samples in (t:=tqdm(batch)):
      inputs, targets = load_batch_samples(samples, data, device=device)
      outputs = model(inputs).squeeze(1)
      loss = F.mse_loss(outputs, targets)
      rloss = rolling_mse_loss(outputs, targets)
      losses.append(loss.item())
      rlosses.append(rloss)
      t.set_description(f'Validating: mse_loss: {loss.item():.2f} | rloss: {rloss:.2f}')
  return torch.tensor(losses), torch.tensor(rlosses)

def train_step(batch, data, epoch=None, val_loss=None):
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
    if epoch > 1:
      scheduler.step(val_loss)
    t.set_description(f'Training: mse_loss: {loss.item():.2f}')
  return torch.tensor(losses)

def train(train_data, val_data):
  tlosses, vlosses = [], []
  val_indexes = load_batch_indexes(np.arange(val_data[0].shape[0]), batch_size=batch_size)
  last = time.monotonic()
  for epoch in range(1, epochs+1):
    val_losses, rval_losses = validation_step(val_indexes, val_data)
    train_indexes = load_batch_indexes(np.arange(train_data[0].shape[0]), batch_size=batch_size, shuffle=True)
    train_losses = train_step(train_indexes, train_data, epoch, val_loss=torch.mean(val_losses))
    tlosses.extend(val_losses); vlosses.extend(train_losses)
    epoch_time = time.monotonic() - last     
    save_model(fn=f"models/{model._get_name()}_{TIMESTAMP}_{epoch}.pt")
   
    ## INFO ##
    print(f'INFO: train_loss {torch.mean(train_losses):.2f} | val_loss {torch.mean(val_losses):2f} | rolling val_loss {torch.mean(rval_losses):.2f}\
          \nINFO: train_loss std {torch.std(train_losses):.2f} | val_loss std {torch.std(val_losses):.2f} | rval_loss std {torch.std(rval_losses):.2f}')
    print(f"{model._get_name()} total done in {epoch_time * epochs // 60} min | Now done in {epoch_time * (epochs - epoch) // 60} min"); last = time.monotonic()
    print('#'*120)
    
  import matplotlib.pyplot as plt
  plt.plot(vlosses.cpu().numpy(), label='Validation Loss')
  plt.plot(tlosses.cpu().numpy(), label='Training Loss')
  plt.legend()
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
  files = os.listdir(OUTPUT_PATH)
  np.random.shuffle(files)
  for filename in (t:=tqdm(files)):
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
  X = data[0][samples].to(device)
  Y = data[1][samples].to(device)
  return X, Y

if __name__ == '__main__':
  device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
 
  epochs = 50
  batch_size = 32
  lr = 1e-3

  model = build_model(weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT).to(device)
  if LOAD:
    model_path = sys.argv[-1]
    print('INFO: loading saved model', model_path)
    model.load_state_dict(torch.load(model_path))
 
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=0.1, 
    patience=3,
    verbose=1,
    min_lr=1e-6,
    cooldown=2)

  dataset = load_data()
  run(dataset)
  