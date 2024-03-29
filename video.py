#!/usr/bin/env python3
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from tqdm import tqdm

DISPLAY = os.getenv('DISPLAY') != None
DURATION = 60

def running_mean(x, N):
  return np.convolve(x, np.ones((N,))/N)[(N-1):]

def get_foundation():
  from ultralytics import YOLO
  yolo_infer = YOLO('/tmp/models/yolov8n-seg.pt')
  def foundation(frame):
    results = yolo_infer.predict(frame, verbose=False, device='mps')
    try:
      masks = results[0].masks.data
    except: return frame
    boxes = results[0].boxes.data
    clss = boxes[:, 5]
    class_indices = torch.where((clss==2))
    class_masks = masks[class_indices]
    class_mask = torch.any(class_masks, dim=0).int() * 255
    mask = class_mask.unsqueeze(0)
    mask = mask.expand(3, *mask.shape[1:]).permute(1, 2, 0)
    mask = np.uint8(mask.cpu().numpy())
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    diff = cv2.bitwise_and(frame, frame, mask=mask)
    return diff
  return foundation

def process_frame(frame, bright_factor):
  hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
  hsv_frame[:, :, 0] = hsv_frame[:, :, 0] * bright_factor  
  frm = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
  frm = frm[196:-196, 232:-232]
  frm = cv2.resize(frm, (320, 160), interpolation=cv2.INTER_AREA)
  return frm

def optical_flowdense_farneback(previous_frame, current_frame, grayout=False):
    original_frame = current_frame.copy()
    if grayout:
      previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
      current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.2

    opflow = cv2.calcOpticalFlowFarneback(
      previous_frame, current_frame,  
      flow_mat, image_scale, nb_images, 
      win_size, nb_iterations, deg_expansion, STD, 0)               
                     
    mag, ang = cv2.cartToPolar(opflow[..., 0], opflow[..., 1])  
    hsv = np.zeros((160, 320, 3))
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 1] = cv2.cvtColor(original_frame, cv2.COLOR_RGB2HSV)[:, :, 1]
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow = cv2.cvtColor(np.asarray(hsv, dtype= np.float32), cv2.COLOR_HSV2RGB)
    if DISPLAY:
      cv2.imshow('frame', flow)
      if cv2.waitKey(1) == ord('q'):
        pass
    flow = cv2.normalize(flow, None, norm_type=cv2.NORM_MINMAX)
    return flow

def gen_frames_from_file(filename, foundation):
  cap = cv2.VideoCapture(filename)    
  ret, frame = cap.read()
  previous_frame = process_frame(foundation(frame), bright_factor=0.2*np.random.uniform())
  frms = []
  while cap.isOpened():
    ret, frame = cap.read()
    if ret is None or not ret: break
    current_frame = process_frame(foundation(frame), bright_factor=0.2*np.random.uniform())
    flow = optical_flowdense_farneback(previous_frame, current_frame, grayout=True)
    previous_frame = current_frame.copy()
    frms.append(flow.mean(axis=2))
    frms = frms[-3:] # Last 3 mean flows
    if len(frms) == 3:
      frmsq = cv2.normalize(np.array(frms), None, norm_type=cv2.NORM_MINMAX)
      yield frmsq, frame # Return original frame aswell for imshow

def load_segments_todisk(video_path, labels_path, output_path):
  if not os.path.exists(output_path): os.makedirs(output_path)
  with open(labels_path, 'r') as file: labels = file.readlines() 

  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames_per_segment = int(fps * DURATION)
  counter = 0
  frame_count = 0
  segment_labels = []
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
  while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_count % frames_per_segment == 0:
      if frame_count > 0:
        with open(f'{output_path}/train_{counter}.txt', 'w') as file:
          file.writelines(segment_labels)
        segment_labels = []
        out.release()
      counter += 1
      out = cv2.VideoWriter(f'{output_path}/train_{counter}.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
   
    out.write(frame)    
    if frame_count < len(labels): segment_labels.append(labels[frame_count])
    frame_count += 1
    pbar.update(1)

  if frame_count % frames_per_segment != 0:
    with open(f'{output_path}/train_{counter}.txt', 'w') as file: file.writelines(segment_labels)
    out.release()

  cap.release()
  print(f"Processed {counter} segments.")

if __name__ == '__main__':
  print(f'Processing {DURATION}s segments...')
  load_segments_todisk('data/train.mp4', 'data/train.txt', 'data/segments')
