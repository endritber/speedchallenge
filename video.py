#!/usr/bin/env python3
import os
from tqdm import tqdm
import numpy as np
import cv2
import torch


RESIZE = (320, 160)
DURATION = 60

def get_foundation():
  from ultralytics import YOLO
  yolo_infer = YOLO('/tmp/models/yolov8n-seg.pt')
  def foundation(frms):
    results = yolo_infer.predict(frms, verbose=False, device='mps') # MPS to make it faster
    try:
      masks = results[0].masks.data
    except:
      return frms
    boxes = results[0].boxes.data
    clss = boxes[:, 5]
    class_indices = torch.where((clss==2) | (clss==3) | (clss==5) | (clss==7)) # car, motorcycle, truck, bus
    class_masks = masks[class_indices]
    class_mask = torch.any(class_masks, dim=0).int() * 255
    mask = class_mask.unsqueeze(0)
    mask = mask.expand(3, *mask.shape[1:]).permute(1, 2, 0)
    diff = cv2.subtract(frms, np.uint8(mask.cpu().numpy()))
    return diff
  return foundation

def process_frame(frm):
  frm = frm.mean(axis=2)/256.
  frm = frm[196:-196, 232:-232]
  frm = cv2.resize(frm, RESIZE)
  return frm

def gen_frames_from_file(filename, foundation):
  cap = cv2.VideoCapture(filename)
  count = 0
  frms = []
  while cap.isOpened():
    ret, frame = cap.read()
    if ret is None or not ret: break
    frm = foundation(frame)
    frm = process_frame(frm)
    frms.append(frm)
    frms = frms[-3:] # keep last 3
    if len(frms) == 3:
      frmsq = np.array(frms).transpose(1, 2, 0)
      yield frmsq
      count+=1

def split_into_segments(video_path, labels_path, output_path):
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  with open(labels_path, 'r') as file:
    labels = file.readlines()

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
    if not ret:
      break

    if frame_count % frames_per_segment == 0:
      if frame_count > 0:
        with open(f'{output_path}/train_{counter}.txt', 'w') as file:
          file.writelines(segment_labels[2:]) # Don't include the first two
        segment_labels = []
        out.release()

      counter += 1
      out = cv2.VideoWriter(f'{output_path}/train_{counter}.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
    
    out.write(frame)    
    if frame_count < len(labels):
      segment_labels.append(labels[frame_count])
    frame_count += 1
    pbar.update(1)

  if frame_count % frames_per_segment != 0:
    with open(f'{output_path}/train_{counter}.txt', 'w') as file:
      file.writelines(segment_labels)
    out.release()

  cap.release()
  print(f"Processed {counter} segments.")

if __name__ == '__main__':
  print(f'Processing {DURATION}s segments...')
  split_into_segments('data/train.mp4', 'data/train.txt', 'data/segments')
