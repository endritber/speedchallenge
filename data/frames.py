import os
import csv
from tqdm import tqdm
import numpy as np
import cv2

DIR = 'comma'
TEST = os.getenv("TEST") != None
 
def save():
  if not TEST:
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

  frame_paths = []
  t = 1
  cap = cv2.VideoCapture(f'{SUBDIR}.mp4')
  ret, frame1 = cap.read()
  prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[..., 1] = 255
  pbar = tqdm(total=pbar)
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    frame = cv2.resize(bgr, (180, 360), interpolation=cv2.INTER_AREA)

    path = DIR+"/"+SUBDIR + f"/frame_{t}.jpg"
    cv2.imwrite(path, frame)
    if type(labels) == np.ndarray: label = labels[t]
    else: label = None
    frame_paths.append({"path": "data/"+path, "label": label})
    t+=1
    pbar.update(1)

  pbar.close()
  #create metadata
  with open(DIR+f'/metadata_{SUBDIR}.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['path', 'label'])
    writer.writerows(frame_paths)

if __name__ == '__main__':
  save()