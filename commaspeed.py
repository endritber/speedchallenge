import os
import time
import csv
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from tqdm.auto import tqdm

DATASET = "data/comma/"

def fetch_metadata(path):
  data = []
  with open(os.path.join(DATASET+"metadata_"+path+".csv"), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      speed = np.float32(row[1])
      data.append((row[0], speed))
  print(f"got {path} metadata", len(data))
  return data

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((80, 160)),
  transforms.Normalize((0.5), (0.5)),
  # transforms.Normalize(
  #      mean=[0.485, 0.456, 0.406],
  #      std=[0.229, 0.224, 0.225]
  #  )
])
def transform_frame(x):
  frame = Image.open(x)
  transformed_frame = transform(frame)
  return transformed_frame

cache = {}
class Comma(Dataset):
  def __init__(self):
    self.meta = fetch_metadata('train')
  
  def __len__(self):
    return len(self.meta)

  def __getitem__(self, idx):
    if idx not in cache:
      x, y = self.meta[idx]
      cache[idx] = transform_frame(x), y
      return cache[idx]

class Rec(nn.Module):
  def __init__(self):
    super().__init__()
    H = 256
    self.prepare = nn.Sequential(
      nn.Linear(160, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU()
    )
    self.encoder = nn.GRU(H, H, batch_first=True)
    self.decode = nn.Sequential(
      nn.Linear(H, H//2),
      nn.ReLU(),
      nn.Linear(H//2, 1)
    )

  def forward(self, x):
    x = self.prepare(x)
    x = x.squeeze(1)
    x = nn.functional.relu(self.encoder(x)[0][:, -1])
    x = self.decode(x)
    return x

def fetch_loader(batch_size):
  dataset = Comma()
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  return dataset, trainloader

if __name__ == '__main__':
  mps_device = torch.device('mps')

  timestamp = time.time()
  epochs = 100
  batch_size = 64
  learning_rate = 0.01
  dataset, trainloader = fetch_loader(batch_size=batch_size)
  
  mse_loss = nn.MSELoss()
  model = Rec().to(mps_device)
  model.load_state_dict(torch.load('models/commaspeed1675506016_3.pt'))
  #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    t = tqdm(trainloader, total=len(dataset)//batch_size)
    model.train()
    for data in t:
      input, target = data
      input = input.to(mps_device)
      target = target.to(mps_device)
      optimizer.zero_grad()
      yhat = model(input)
      loss = mse_loss(yhat, target.unsqueeze(1))
      #print(yhat[0], target.unsqueeze(1)[0])
      loss.backward()
      optimizer.step()
      t.set_description(f"loss {loss.item()}")

    torch.save(model.state_dict(), f'models/commaspeed{int(timestamp)}_{epoch}.pt')
