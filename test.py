#!/usr/bin/env python3
import numpy as np
import cv2
import torch
from train import build_model
from video import get_foundation, gen_frames_from_file

WINDOWS = 10

def running_mean(x, N):
  return np.convolve(x, np.ones((N,))/N)[(N-1):]

if __name__ == '__main__':
  device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
  model = build_model(fine_tune=False).to(device)
  model.load_state_dict(torch.load("models/EfficientNet_1699123751_11.pt"))
  model.eval()
  foundation = get_foundation()
  outs = []
  for frm, original_frame in gen_frames_from_file('data/test.mp4', foundation):  
    out = model(torch.tensor(frm, dtype=torch.float32).to(device).unsqueeze(0)).cpu().item()
    outs.append(out)
    for j in range(1, 3, 1):
      out_to_show = running_mean(np.array(outs[-(WINDOWS-1):]), j)  
    out_to_show = out_to_show.mean()
    cv2.putText(original_frame, 'Predicted Speed: ' + str(round(out_to_show, 1)) + ' m/h', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (124,252,0), 2)
    cv2.imshow('frame', original_frame)
    if cv2.waitKey(1) == ord('q'):
        break
