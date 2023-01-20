import cv2
import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(sci_mode=True)
from tqdm import tqdm

path = 'data/train.mp4'
cap = cv2.VideoCapture(path)


pbar = tqdm(total=20400)
frames = []
while 1:
  ret, frame = cap.read()
  if not ret:
    break

  frame = frame.mean(axis=2)/256.0
  frame = cv2.resize(frame, (320, 160))
  frames.append(frame)
  pbar.update(1)

cap.release()
pbar.close()

x_train = np.array(frames, dtype=np.float32)
with open('data/train.txt') as f:
  y_train = f.readlines()

y_train = np.array(y_train, dtype=np.float32)

print('loaded', len(x_train), 'frames...')

#model
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(320*160, 512),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(512, 1)
    )

  def forward(self, x):
    out = self.layers(x)
    return out

device = torch.device('mps')
model = MLP().to(device)

criterion = nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#train
epochs = 10
for epoch in range(epochs):

  divider = int(len(x_train)*0.9)

  x_train_data = torch.from_numpy(x_train[:divider]).to(device)
  y_train_data = torch.from_numpy(y_train[:divider]).to(device)
  
  x_val_data = torch.from_numpy(x_train[divider:]).to(device)
  y_val_data = torch.from_numpy(y_train[divider:]).to(device)

  for train_data in zip(x_train_data, y_train_data):
    
    inputs, targets = train_data
    inputs = inputs.reshape(1, -1)
    targets = targets.reshape(-1, 1)

    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()
 
  for val_data in zip(x_val_data, y_val_data):
    inputs_val, targets_val = val_data
    inputs_val = inputs_val.reshape(1, -1)
    targets_val = targets_val.reshape(-1, 1)

    outputs_val = model(inputs_val)
    loss_val = criterion(outputs_val, targets_val)

  print('Epoch [{}/{}], Train Loss: {}, Val Loss {}'.format(epoch+1, epochs, loss.item(), loss_val.item()))

torch.save(model.state_dict(), 'model/mlp.pt')
