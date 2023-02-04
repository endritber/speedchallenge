import os
import time
import csv
import wandb
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

def fetch_metadata(path, validation):
  data = []
  with open(os.path.join(DATASET+"metadata_"+path+".csv"), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      speed = np.float32(row[1])
      data.append((row[0], speed))

  n = len(data)
  split = int(n*0.8)
  if validation:
    data = data[split:]
  else:
    data = data[:split]

  print(f"got metadata", len(data))
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
  def __init__(self, validation):
    self.meta = fetch_metadata('train', validation)


  def __len__(self):
    return len(self.meta)

  def __getitem__(self, idx):
    if idx not in cache:
      x, y = self.meta[idx]
      cache[idx] = transform_frame(x), y
      return cache[idx]

class TemporalBatchNorm(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.bn = nn.BatchNorm1d(channels)

  def forward(self, x):
    # (N, C, H, W)
    x = x.squeeze(1)
    xx = x.permute(0, 2, 1)
    xx = self.bn(xx)
    xx = xx.permute(0, 1, 2)
    # (N , H, W)
    return x

class Rec(nn.Module):
  def __init__(self):
    super().__init__()
    H = 360
    self.prepare = nn.Sequential(
      nn.Linear(160, H),
      TemporalBatchNorm(H),
      nn.ReLU(),
      nn.Linear(H, H),
      TemporalBatchNorm(H),
      nn.ReLU(),
      nn.Linear(H, H),
      TemporalBatchNorm(H),
      nn.ReLU()
    )
    self.encoder = nn.GRU(H, H, batch_first=True)
    self.decode = nn.Sequential(
      nn.Linear(H, H//2),
      TemporalBatchNorm(H//2),
      nn.ReLU(),
      nn.Linear(H//2, H//4),
      TemporalBatchNorm(H//4),
      nn.ReLU(),
      nn.Linear(H//4, 1)
    )

  def forward(self, x):
    x = self.prepare(x)
    x = x.squeeze(1)
    x = self.encoder(x)[0]
    x = self.decode(x)[:, -1]
    return x

def fetch_loader(batch_size, validation):
  dataset = Comma(validation)
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  return dataset, trainloader

if __name__ == '__main__':

  WAN = int(os.getenv('WAN')) != 0

  mps_device = torch.device('mps')
  timestamp = time.time()
  epochs = 300
  batch_size = 16
  learning_rate = 0.001

  if WAN:
    wandb.init(project="test-project", entity="endritber")
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size
    }

  dataset, trainloader = fetch_loader(batch_size=batch_size, validation=False)
  val_dataset, validation_loader = fetch_loader(batch_size=batch_size, validation=True)
  mse_loss = nn.MSELoss()
  model = Rec().to(mps_device)
  #model.load_state_dict(torch.load('models/commaspeed1675510198_2.pt'))
  #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    if WAN:
      wandb.watch(model)

    with torch.no_grad():
      model.eval()
      #evaluate
      losses = []
      for valdata in (t:=tqdm(validation_loader)):
        valinput, valtarget = valdata
        valinput = valinput.to(mps_device)
        valtarget = valtarget.to(mps_device)
        valyhat = model(valinput)
        valloss = mse_loss(valyhat, valtarget.unsqueeze(1))
        t.set_description(f"val loss {valloss.item()}")
        losses.append(valloss)
        if WAN:
          wandb.log({"val_loss": valloss})
      val_loss = torch.mean(torch.tensor(losses)).item()
      print(f"avg val_loss: {val_loss:.2f}")

    model.train()
    for data in (t:=tqdm(trainloader, total=len(dataset)//batch_size)):
      input, target = data
      input = input.to(mps_device)
      target = target.to(mps_device)
      optimizer.zero_grad()
      yhat = model(input)
      loss = mse_loss(yhat, target.unsqueeze(1))
      #print(yhat[0], target.unsqueeze(1)[0])
      # print(yhat.squeeze(1).shape)
      # print(target.shape)
      loss.backward()
      optimizer.step()
      t.set_description(f"loss {loss.item()}")
      if WAN:
        wandb.log({"loss": loss})
        wandb.log({"actualspeed": torch.mean(target)})
        wandb.log({"predictedspeed": torch.mean(yhat.squeeze(1))})


    torch.save(model.state_dict(), f'models/commaspeed{int(timestamp)}_{epoch}.pt')
