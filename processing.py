import os
import csv
import cv2
import functools
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

DATASET = "data/comma/"

@functools.lru_cache(None)
def fetch_metadata(path):
  data = []
  with open(os.path.join(DATASET+"metadata_"+path+".csv"), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      speed = np.float32(row[1])
      data.append((row[0], speed))
  return data

def read_frame(x):
  frame = cv2.imread(x)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).transpose(1, 0)
  print(frame.shape)
  tensor = torch.tensor(frame, dtype=torch.float32)
  return tensor

if __name__ == "__main__":

  TEST = os.getenv('TEST') != None

  if not False:
    TEST = 'test'
  else:
    TEST = 'train'

  meta = fetch_metadata('train')
  x, y = [], []
  for frame, label in tqdm(meta):
    x.append(read_frame(frame))
    y.append(label)

  x = torch.stack(x)
  x = x.to(torch.float32)

  print('saving...')
  torch.save(x, "data/comma_speed_x.pt")
  torch.save(torch.tensor(y, dtype=torch.float32), "data/comma_speed_y.pt")