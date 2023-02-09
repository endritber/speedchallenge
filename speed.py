import os
import time
import shutil
import numpy as np
import cv2
from preprocessing import calculate_opticalflow
from model import SpeedResNet34, NVidiaNet
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
  # model = SpeedResNet34().to(device)
  # model.load_state_dict(torch.load("demo/speedresnet34_1675938944_53.pt")) 
  model = NVidiaNet().to(device)
  model.load_state_dict(torch.load("demo/nvidianet_1675948200_15.pt"))
  model.eval()
  
  cap = cv2.VideoCapture(PATH)
  previous_frame = cap.read()[1]

  i = 0
  if os.path.exists('/tmp/comma/speed/test'):
    shutil.rmtree('/tmp/comma/speed/test')
  os.mkdir('/tmp/comma/speed/test') 
  while True:
    previous_frame = cap.read()[1]
    ret, current_frame = cap.read()
    if not ret:
      print('cannot open frame (stream end?)')
      break

    previous_frame_path = f'/tmp/comma/speed/test/previous_frame_{i}.jpg' 
    current_frame_path = f'/tmp/comma/speed/test/current_frame_{i}.jpg'
    cv2.imwrite(previous_frame_path, previous_frame)
    cv2.imwrite(current_frame_path, current_frame)

    flow = calculate_opticalflow(previous_frame_path, current_frame_path, augmentation=False).transpose(2, 0, 1)
    print(flow)
    out = model(torch.tensor(flow, dtype=torch.float32)[None, :].to(device))
    
    os.remove(previous_frame_path)
    os.remove(current_frame_path)
    
    if SHOW:
      time.sleep(0.03)
      out = out.cpu().detach().numpy()[0, 0]
      if out < 0:
        out = 0.00
      cv2.putText(previous_frame, 'Predicted Speed: ' + str(round(out, 2)) + ' mph', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (124,252,0), 2)	
      if not TEST:
        cv2.putText(previous_frame, 'Actual Speed: ' + str(round(speed[i], 2)) + ' mph', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)	

      cv2.imshow('model testing (press q to quit)', previous_frame)  
      i+=1
      if cv2.waitKey(1) == ord('q'):
        break
  
  cap.release()
  cv2.destroyAllWindows()