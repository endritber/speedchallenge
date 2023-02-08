import os
import shutil
import time
from tqdm.auto import tqdm
import csv
import cv2 
import numpy as np


path='train'
DATA_PATH = f'data/{path}.mp4'
COMMA_SPEED_PATH = '/tmp/comma/speed'

def create_directory():
  print(f"creating directory 'speed' in {COMMA_SPEED_PATH[:-6]}")
  os.mkdir(f'{COMMA_SPEED_PATH}')
  os.mkdir(f'{COMMA_SPEED_PATH}/{path}') 
 
def load_todisk():
  if os.path.exists(COMMA_SPEED_PATH):
      print(COMMA_SPEED_PATH, 'exists')
      print('removing...')
      shutil.rmtree(COMMA_SPEED_PATH)
      create_directory()
  else:
      create_directory()

  cap = cv2.VideoCapture(DATA_PATH)
  if not cap.isOpened():
    print('cannot find video file in this path', DATA_PATH)
    exit()
  speed = np.loadtxt(DATA_PATH[:-4]+'.txt', dtype=np.float32)
  metadata = open(f'data/metadata_{path}.csv', 'w')
  writer = csv.writer(metadata)
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  pbar = tqdm(total=n)
  c = 0
  while True:
    frame_time_now = time.monotonic()
    ret, frame = cap.read()
    if not ret:
      print('cannot read frame') 
      continue
      
    frame_path = os.path.join(COMMA_SPEED_PATH, path, f'frame{frame_time_now:.5f}.jpg')
    cv2.imwrite(frame_path, frame)
    writer.writerow([frame_path, speed[c]])
    
    c+=1
    pbar.update(1)
    
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  load_todisk()

