import os
import csv
from tqdm import tqdm
import numpy as np
import cv2

DIR = 'comma'
TRAIN = int(os.getenv("TRAIN"))
 
def save():
  if TRAIN == None:
    print("set TRAIN=1 or TRAIN=0")
    exit()

  if TRAIN:
    SUBDIR = 'train'
    pbar = 20400
    labels = np.loadtxt('train.txt', dtype=np.float32)
  else:
    SUBDIR = 'test'
    pbar = 10798
    labels = None

  try:
    if not os.path.exists(DIR):
      os.mkdir(DIR)
    saving_path = os.path.join(DIR, SUBDIR)
    if not os.path.exists(saving_path):
      os.mkdir(saving_path)
  except FileExistsError:
    pass

  frames = []
  t = 0
  cap = cv2.VideoCapture(f'{SUBDIR}.mp4')
  pbar = tqdm(total=pbar)
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    path = DIR+"/"+SUBDIR + f"/frame{t+1}.jpg"
    cv2.imwrite(path, frame)
    if type(labels) == np.ndarray: label = labels[t]
    else: label = None
    frames.append({"path": path, "label": label})
    t+=1
    pbar.update(1)

  pbar.close()
  #create metadata
  with open(DIR+f'/metadata_{SUBDIR}.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["path", "label"])
    writer.writeheader()
    writer.writerows(frames)

if __name__ == '__main__':
  save()