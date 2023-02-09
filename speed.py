import os
import time
import numpy as np
np.set_printoptions(suppress=True)
import cv2
from preprocessing import calculate_opticalflow
from model import SpeedNet
import torch

SHOW = os.getenv('SHOW') != None
TEST = os.getenv('TEST') != None

if TEST:
   data_path = 'test'
   speed = None
else:
   data_path = 'train'
   speed = np.loadtxt('data/train.txt')

PATH = f'data/{data_path}.mp4'

if __name__ == '__main__':

  device = 'mps' if torch.backends.mps.is_available() else 'cpu'
  model = SpeedNet().to(device)
  model.eval()
  model.load_state_dict(torch.load("demo/"))

  cap = cv2.VideoCapture(PATH)
  previous_frame = cap.read()[1]
  if not cap.isOpened():
    print('cannot open video')
    exit()

  i = 0
  while True:
    ret, current_frame = cap.read()
    if not ret:
      print('cannot open frame (stream end?)')
      break

    flow = calculate_opticalflow(previous_frame, current_frame, augmentation=False)
    print(flow)
    out = model(torch.tensor(flow, dtype=torch.float32)[None, :].to(device))
    previous_frame = current_frame
    
    if SHOW:
      out = out.cpu().detach().numpy()[0, 0]
      if out < 0:
        out = 0.00
      cv2.putText(current_frame, 'Predicted Speed: ' + str(round(out, 2)) + ' mph', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (233,150,122), 2)	
      if not TEST:
        cv2.putText(current_frame, 'Actual Speed: ' + str(round(speed[i], 2)) + ' mph', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)	
        
      cv2.imshow('model testing (press q to quit)', current_frame)  
      i+=1
      if cv2.waitKey(1) == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()
