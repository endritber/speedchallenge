from PIL import Image, ImageEnhance
import numpy as np
import cv2

C, W, H = 3, 160, 50
RESIZE = (W, H)
HSV = np.zeros((H, W, C), dtype=np.float32)
HSV[..., 1] = 255

def process_frames(current_frame, next_frame, augmentation):
  current_frame = Image.open(current_frame).crop((0, 170, 640, 370)).resize(RESIZE)
  next_frame = Image.open(next_frame).crop((0, 170, 640, 370)).resize(RESIZE)
  
  if augmentation:
    brightness = np.random.uniform(0.2, 2)
    current_frame = ImageEnhance.Brightness(current_frame).enhance(brightness)
    next_frame = ImageEnhance.Brightness(next_frame).enhance(brightness) 
    color = np.random.uniform(0.2, 2)
    current_frame = ImageEnhance.Brightness(current_frame).enhance(color)
    next_frame = ImageEnhance.Brightness(next_frame).enhance(color) 
    if np.random.uniform(0, 1) < 0.5:
      current_frame = current_frame.transpose(Image.FLIP_LEFT_RIGHT)
      next_frame = next_frame.transpose(Image.FLIP_LEFT_RIGHT)

  current_frame = cv2.cvtColor(np.asarray(current_frame, dtype=np.float32), cv2.COLOR_BGR2GRAY)
  next_frame = cv2.cvtColor(np.asarray(next_frame, dtype=np.float32), cv2.COLOR_BGR2GRAY)
  return current_frame, next_frame

def calculate_opticalflow(current_frame, next_frame, augmentation=True):
  current_frame, next_frame = process_frames(current_frame, next_frame, augmentation)
  flow = cv2.calcOpticalFlowFarneback(current_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
  HSV[..., 0] = angle*180/np.pi/2
  HSV[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
  #print(HSV.shape)
  flow = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
  return flow