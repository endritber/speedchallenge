import csv
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from preprocessing import calculate_opticalflow

def load_metadata(path):
  data = []
  with open(path, newline='') as csvfile:
    reader = list(csv.reader(csvfile, delimiter=','))
    for i in range(len(reader)):
      current_frame = reader[i][0]
      try:
        next_frame = reader[i+1][0]
      except:
        next_frame = current_frame
      speed = reader[i][1]

      data.append((current_frame, next_frame, np.float32(speed)))
  return data

transform=ToTensor()
class CommaSpeedDataset(Dataset):
  def __init__(self, path, augmentation=True, validation=False):
    self.augmentation = augmentation
    self.metadata = load_metadata(path)
    self.metadata = pd.DataFrame(self.metadata, columns=['curr_frame', 'next_frame', 'speed'])
    trainval = self.metadata.iloc[:-1200]
    self.metadata = trainval.sample(frac=0.8,random_state=200)

    if validation:
      self.metadata = trainval.drop(self.metadata.index)

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, index):
    row = self.metadata.iloc[index]
    pf = cv2.imread(row['curr_frame'])
    nf = cv2.imread(row['next_frame'])
    flow = calculate_opticalflow(pf, nf, augmentation=self.augmentation)
    return transform(flow), row['speed']
  
if __name__ == '__main__':
  dataset = CommaSpeedDataset('data/metadata_train.csv')
  for i, y in dataset:
    print(i.shape, y)
  
