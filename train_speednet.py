import time
from tqdm import tqdm
from dataset import CommaSpeedDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

class SpeedNet(nn.Module):
  def __init__(self):
    super(SpeedNet, self).__init__()
    resnet = list(resnet101(weights=ResNet101_Weights.DEFAULT).children())[:-4] #except last 2 layers    
    self.conv_seq1 = nn.Sequential(
      nn.Conv2d(3, 3, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False),
      nn.BatchNorm2d(3),
      nn.ELU())
    self.resnet = nn.Sequential(*resnet)
    self.conv_seq2 = nn.Sequential(
      nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
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

# SpeedNet()(torch.zeros((64, 3, 50, 160)))
# exit()

if __name__ == '__main__':
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'

  comma_speed_dataset_train = CommaSpeedDataset('data/metadata_train.csv', augmentation=True)
  comma_speed_dataset_val = CommaSpeedDataset('data/metadata_train.csv', augmentation=False, validation=True)

  epochs = 60
  batch_size = 32
  learning_rate = 0.001

  train_dataloader = DataLoader(comma_speed_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
  val_dataloader = DataLoader(comma_speed_dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)

  model = SpeedNet().to(device)
  print(model)
  #model.load_state_dict(torch.load("model/speednet_1675877563_24.pt"))
  #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    timestamp = time.time()

    val_losses = []
    model.eval()
    print('VALIDATION')
    for val_input, val_target in (t:=tqdm(val_dataloader)):
      val_input = val_input.to(device)
      val_target = val_target.to(device)
      val_out = model(val_input)
      val_loss = F.mse_loss(val_out, val_target.unsqueeze(1)).to(device)
      t.set_description(f"val loss {val_loss.item():.2f}")
      val_losses.append(val_loss.item())
    print('Validation Loss', torch.tensor(val_losses).mean())

    train_losses = []
    model.train()
    for input, target in (t:=tqdm(train_dataloader)):
      input = input.to(device)
      target = target.to(device)
      optimizer.zero_grad()
      out = model(input)
      #print(out.shape, target.unsqueeze(1).shape)
      loss = F.mse_loss(out, target.unsqueeze(1))
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
      t.set_description(f"loss {loss.item():.2f}")
    print('Train Loss', torch.tensor(train_losses).mean())
    
    torch.save(model.state_dict(), f'model/speednet_{int(timestamp)}_{epoch}.pt')
