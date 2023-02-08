import csv
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from preprocessing import calculate_opticalflow

def load_metadata(path):
  data = []
  with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      data.append((row[0], np.float32(row[1])))
  return data

transform=ToTensor()
class CommaSpeedDataset(Dataset):
  def __init__(self, path, augmentation=True, validation=False):
    self.augmentation = augmentation
    self.metadata = load_metadata(path)

    split = 0.85 * len(self.metadata)
    if not validation:
      self.metadata = self.metadata[:int(split)]
    else:
      self.metadata = self.metadata[int(split):]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, index):
    previous_frame, speed = self.metadata[index]
    try:
      next_frame = self.metadata[index+1][0]
    except:
      next_frame = previous_frame

    pf = cv2.imread(previous_frame)
    nf = cv2.imread(next_frame)
    flow = calculate_opticalflow(pf, nf, augmentation=self.augmentation)
    return transform(flow), speed
  
if __name__ == '__main__':
  dataset = CommaSpeedDataset('data/metadata_train.csv')
  for i, y in dataset:
    print(i.shape, y)
  
