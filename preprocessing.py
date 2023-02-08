from PIL import Image, ImageEnhance
import numpy as np
import cv2

C, W, H = 3, 160, 50
RESIZE = (W, H)
HSV = np.zeros((H, W, C), dtype=np.float32)
HSV[..., 1] = 255

def augment(frame):
  frame = Image.fromarray(frame)
  brightness = np.random.uniform(0.2, 2)
  frame = ImageEnhance.Brightness(frame).enhance(brightness)
  color = np.random.uniform(0.2, 2)
  frame = ImageEnhance.Brightness(frame).enhance(color)
  if np.random.uniform(0, 1) < 0.5:
    frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
  frame = np.asarray(frame, dtype=np.float32)
  return frame

def process_frame(frame):
  frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), RESIZE)
  return frame

def calculate_opticalflow(previous_frame, current_frame, augmentation=True):
  if augmentation:
    previous_frame, current_frame = augment(previous_frame), augment(current_frame)

  previous_frame, current_frame = process_frame(previous_frame), process_frame(current_frame)
  flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
  HSV[..., 0] = angle*180/np.pi/2
  HSV[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
  #print(HSV.shape)
  flow = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
  return flow