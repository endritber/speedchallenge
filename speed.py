import os
import numpy as np
np.set_printoptions(suppress=True)
import cv2
from preprocessing import calculate_opticalflow

SHOW = os.getenv('SHOW') != None
TEST = os.getenv('TEST') != None

if TEST: data_path = 'test'
else: data_path = 'train'

PATH = f'data/{data_path}.mp4'

if __name__ == '__main__':
  cap = cv2.VideoCapture(PATH)
  previous_frame = cap.read()[1]
  if not cap.isOpened():
    print('cannot open video')
    exit()

  while True:
    ret, current_frame = cap.read()
    if not ret:
      print('cannot open frame (stream end?)')
      break

    flow = calculate_opticalflow(previous_frame, current_frame, brightness=False)
    previous_frame = current_frame
    
    if SHOW:
      # cv2.putText(frame, str(speed[i]) + ' mph', (10, 30),
		  # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (, 255, 0), 2)	
      cv2.imshow('frame (press q to quit)', previous_frame)  
      if cv2.waitKey(1) == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()