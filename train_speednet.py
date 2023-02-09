import time
from tqdm import tqdm
from dataset import CommaSpeedDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import SpeedNet, NVidiaNet

# SpeedNet()(torch.zeros((64, 3, 50, 160)))
# exit()

if __name__ == '__main__':
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'

  comma_speed_dataset_train = CommaSpeedDataset('data/metadata_train.csv', augmentation=True)
  comma_speed_dataset_val = CommaSpeedDataset('data/metadata_train.csv', augmentation=False, validation=True)

  epochs = 100
  batch_size = 32
  learning_rate = 0.0001

  train_dataloader = DataLoader(comma_speed_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
  val_dataloader = DataLoader(comma_speed_dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

  #model = SpeedNet().to(device)
  #model.load_state_dict(torch.load("demo/speednet_1675938944_53.pt"))
  model = NVidiaNet().to(device)
  
  #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(1, epochs+1):
    timestamp = time.time()

    val_losses = []
    model.eval()
    print('#######################################################\nMODEL VALIDATION')
    for val_input, val_target in (t:=tqdm(val_dataloader)):
      val_input = val_input.to(device)
      val_target = val_target.to(device)
      val_out = model(val_input)
      val_loss = F.mse_loss(val_out, val_target.unsqueeze(1))
      t.set_description(f"val loss {val_loss.item():.2f}")
      val_losses.append(val_loss.item())
    print(f'val loss:', torch.tensor(val_losses).mean().item())
    print("#######################################################\nMODEL TRAINING")

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
      t.set_description(f"epoch {epoch}: train loss {loss.item():.2f}")
    print(f'epoch {epoch} -> train loss:', torch.tensor(train_losses).mean().item())
    
    torch.save(model.state_dict(), f'models/nvidianet_{int(timestamp)}_{epoch}.pt')
