import os
import time
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WAN = os.getenv('WAN') != None
GPU = os.getenv('GPU') != None

if GPU:
  DEVICE = 'mps'
else:
  DEVICE = 'cpu'

DEVICE = torch.device(DEVICE)

def load_examples():
  global X, y
  print('loading X data')
  X = torch.load('data/comma_speed_x.pt')
  X = X.squeeze(1)
  if GPU:
    print('copying to gpu', X.shape)
    X = X.to(device='mps', non_blocking=True)
  print('loading Y data')
  y = torch.load('data/comma_speed_y.pt')
  print('data_loaded')

def get_sample(samples):
  input = X[samples]
  target = y[samples]
  return input, target

class TemporalBatchNorm(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.bn = nn.BatchNorm1d(channels)

  def forward(self, x):
    # (N, C, H, W)
    x = x.squeeze(1)
    xx = x.permute(0, 2, 1)
    xx = self.bn(xx)

    # (N , H, W)
    return x

class ResBlock(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c),
      nn.ReLU(),
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c),
    )

  def forward(self, x):
    return nn.functional.relu(x+self.block(x))


class Rec(nn.Module):
  def __init__(self):
    super().__init__()

    C=16
    self.encode = nn.Sequential(
      nn.Conv2d(1, C, 3, stride=(1, C//2), padding=(1, 0)),
      ResBlock(C),
      nn.MaxPool2d((2, 2)),
      ResBlock(C),
      nn.MaxPool2d((2, 2)),
      ResBlock(C),
    )
    H = 180
    self.flatten = nn.Linear(176, H)
    self.gru = nn.GRU(H, H, batch_first=True)
    self.decode = nn.Sequential(
      nn.Linear(45*180, H//2),
      nn.BatchNorm1d(H//2),
      nn.ReLU(),
      nn.Linear(H//2, 1))

  def forward(self, x):
    x = self.encode(x[:, None]) # (batch, C, H, W)
    x = x.permute(2, 0, 1, 3) # (H, batch, C, W)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = self.flatten(x)
    x = self.gru(x)[0]
    x = x.permute(1, 0, 2)
    x = torch.flatten(x, start_dim=1)
    x = self.decode(x)
    return x


def train():
  timestamp = time.time()
  epochs = 100
  batch_size = 64
  learning_rate = 0.001

  if WAN:
    wandb.init(project="test-project", entity="endritber")
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size
    }

  mse_loss = nn.MSELoss().to(DEVICE)
  model = Rec().to(DEVICE)  
  #model.load_state_dict(torch.load('models/commaspeed1675678052_1.pt'))
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  split = int(X.shape[0]*0.9)
  trains = [x for x in list(range(split))*4]

  for epoch in range(epochs):
    if WAN:
      wandb.watch(model)

    # with torch.no_grad():
    #   model.eval()
    #   #evaluate
    #   val_losses = []
    #   for valdata in (t:=tqdm(validation_loader)):
    #     valinput, valtarget = valdata
    #     valinput = valinput.to(mps_device)
    #     valtarget = valtarget.to(mps_device)
    #     valyhat = model(valinput)
    #     valloss = mse_loss(valyhat, valtarget.unsqueeze(1))
    #     t.set_description(f"val loss {valloss.item():.2f}")
    #     val_losses.append(valloss)
    #     if WAN:
    #       wandb.log({"val_loss": valloss})
    #       wandb.log({"val_actual_speed": torch.mean(valtarget)})
    #       wandb.log({"val_predicted_speed": torch.mean(valyhat.squeeze(1))})
    #   val_loss = torch.mean(torch.tensor(val_losses)).item()
    #   print(f"avg val_loss: {val_loss:.2f}")

    model.train()
    #random.shuffle(trains)
    batches = np.array(trains)[:len(trains)//batch_size * batch_size].reshape(-1, batch_size)

    for sample in (t:=tqdm(batches)):
      input, target = get_sample(sample)
      target = target.to(DEVICE)
      optimizer.zero_grad()
      out = model(input)
      loss = mse_loss(out, target.unsqueeze(1))
      loss.backward()
      optimizer.step()
      t.set_description(f"loss {loss.item():.2f}")
      if WAN:
        wandb.log({"loss": loss})
      
    torch.save(model.state_dict(), f'models/commaspeed{int(timestamp)}_{epoch}.pt')

if __name__ == '__main__':
  load_examples()
  train()
  