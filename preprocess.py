#!/usr/bin/env python3
import cv2
import numpy as np
import torch

if __name__ == '__main__':
  filename = 'data/train.mp4'
  cap = cv2.VideoCapture(filename)
  frms, xs = [], []
  while cap.isOpened():
    ret, frm = cap.read()
    if not ret: break
    frms.append(frm)

  if frms:
    cap.release()
    big_data = torch.Tensor(np.concatenate([x.reshape(1,480,640,3) for x in frms]))
    print(big_data.shape)
    torch.save(big_data, 'data/train.pt')
    