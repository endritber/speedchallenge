import cv2
import torch
mps_device = torch.device("mps")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from tqdm.auto import tqdm

STREAM = 'data/train.mp4'

def load_stream_frames(max_frames=20400):
  print('loading data')
  cap = cv2.VideoCapture(STREAM)
  if not cap.isOpened():
    print('cannot open stream')
    exit()

  with open(STREAM[:-4]+'.txt') as file:
    labels = file.readlines()

  pbar = tqdm(total=max_frames)
  idx = 0
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    
    frame = frame.mean(axis=2)/255.0
    frame = cv2.resize(frame, (256, 128))
    frames.append([frame, float(labels[idx])])
    idx+=1
    pbar.update(1)
    if max_frames == idx+1:
      break

  pbar.close()
  print('loaded data')
  return frames

class CommaData(Dataset):
  def __init__(self, max_frames, val=False):
    self.data = load_stream_frames(max_frames=max_frames)
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
      x, y = self.data[idx]
      return x, y

class ResBlock(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c),
      nn.ReLU(c),
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c))

  def forward(self, x):
    return nn.functional.relu(x + self.block(x))

class Rec(nn.Module):
  def __init__(self):
    super().__init__()
    self.encode = nn.Sequential(
      nn.Conv2d(1, 16, 3),
      ResBlock(16),
      ResBlock(16),
      ResBlock(16),
    )
    self.flatten = nn.Linear(4064, 128)
    self.gru = nn.GRU(128, 128, batch_first=True)
    self.decode = nn.Sequential(
       nn.Linear(128, 64),
       nn.BatchNorm1d(64),
       nn.ReLU(),
       nn.Dropout(0.5),
       nn.Linear(64, 1))

  def forward(self, x):
    # (batch, C, H, W)
    x = self.encode(x).permute(0, 2, 1, 3) 
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = self.flatten(x)
    x = self.gru(x)[0][:, -1]
    x = self.decode(x)
    return x

def get_loaders(max_frames, batch_size):
  dataset = CommaData(max_frames)
  # validation_split = 0.1
  # indices = list(range(len(dataset)))
  # split = int(np.floor(validation_split * len(dataset)))
  # train_indices, val_indices = indices[split:], indices[:split]
  
  # train_sampler = SequentialSampler(train_indices)
  # valid_sampler = SequentialSampler(val_indices)

  trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                shuffle=True, num_workers=4)

  # valloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
  #               shuffle=False, num_workers=4, sampler=valid_sampler)

  return dataset, trainloader

if __name__ == '__main__':
  import time
  batch_size = 32
  #frames = 20400
  frames = 16700
  start = time.time()
  dataset, trainloader = get_loaders(max_frames=frames, batch_size=batch_size)
  print(time.time()-start, 'data_loaders')
  mse_loss = nn.MSELoss()
  model = Rec().to(mps_device)
  #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  for epoch in range(100):
    start = time.time()
    t = tqdm(trainloader, total=len(dataset)//batch_size)
    print(time.time()-start, 'tqdm')
    model.train()
    for data in t:
      input, target = data
      input = input.to(torch.float32).to(mps_device)[:, None, :, :]
      target = target.to(torch.float32).to(mps_device)
      optimizer.zero_grad()
      yhat = model(input)
      loss = mse_loss(yhat, target.unsqueeze(-1))
      loss.backward()
      optimizer.step()
      print(loss.item())
      t.set_description(f"loss {loss.item()}")

    # with torch.no_grad():
    #   model.eval()
    #   running_vloss = 0.0
    #   for i, vdata in enumerate(valloader):
    #       vinputs, vlabels = vdata
    #       vinputs = vinputs.to(torch.float32).to(mps_device)
    #       vlabels = vlabels.to(torch.float32).to(mps_device)
    #       voutputs = model(vinputs)
    #       vloss = mse_loss(voutputs, vlabels.unsqueeze(-1))
    #       running_vloss += vloss

    # avg_vloss = running_vloss / (i + 1)
    # print('train loss {} validation loss {}'.format(loss.item(), avg_vloss))
