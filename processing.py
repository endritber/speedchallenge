import os
import csv
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

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((180, 360)),
  transforms.Normalize((0.5), (0.5)),
])
def transform_frame(x):
  frame = Image.open(x)
  transformed_frame = transform(frame)/256.0
  return transformed_frame

if __name__ == "__main__":
  meta = fetch_metadata('train')
  x, y = [], []
  for frame, label in tqdm(meta):
    x.append(transform_frame(frame))
    y.append(label)

  x = torch.stack(x)
  x = x.to(torch.float32)

  print('saving...')
  torch.save(x, "data/comma_speed_x.pt")
  torch.save(torch.tensor(y, dtype=torch.float32), "data/comma_speed_y.pt")