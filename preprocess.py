#!/usr/bin/env python3
import os
import tqdm
import cv2
import numpy as np
import torch

RESIZE = (320, 160)

def get_foundation():
  from ultralytics import YOLO
  yolo_infer = YOLO('yolov8n-seg.pt')
  def foundation(frm):
    results = yolo_infer.predict(frm, verbose=False, device='mps') # MPS to make it faster
    try:
      masks = results[0].masks.data
    except:
      return frm
    boxes = results[0].boxes.data
    clss = boxes[:, 5]
    class_indices = torch.where((clss==2) | (clss==3) | (clss==5) | (clss==7)) # car, motorcycle, truck, bus
    class_masks = masks[class_indices]
    class_mask = torch.any(class_masks, dim=0).int() * 255
    mask = class_mask.unsqueeze(0)
    mask = mask.expand(3, *mask.shape[1:]).permute(1, 2, 0)
    diff = cv2.subtract(frm, np.uint8(mask.cpu().numpy()))
    return diff
  return foundation

def process_frame(frm):
  frm = frm.mean(axis=2)/256.
  frm = frm[196:-196, 232:-232]
  frm = cv2.resize(frm, RESIZE)
  return frm

def gen_frames_fromfile(filename, foundation):
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

def preprocess(foundation, filename, filename_out, transform=None):
  if not filename.endswith(".mp4"): return
  print('preprocessing', filename, filename_out)
  if os.path.isfile(filename_out): return
  frms = []
  for frm in tqdm.tqdm(gen_frames_fromfile(filename, foundation)):
    frms.append(frm)

  if frms:
    big_x = torch.as_tensor(frms)
    print('Saving:', big_x.shape, big_x.dtype)
    torch.save(big_x, filename_out)

if __name__ == '__main__':
  filename = 'data/test.mp4'
  foundation = get_foundation()
  for filename in os.listdir("data"):
    if filename.endswith('.mp4'):
      preprocess(foundation, "data/"+filename, "data/"+filename.replace('.mp4', ".pt"), None)