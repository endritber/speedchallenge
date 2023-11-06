#!/usr/bin/env python3
import os
import numpy as np
import cv2
import torch
from train import build_model
from video import get_foundation, gen_frames_from_file, running_mean

WINDOWS = 25
ROLLING = os.getenv('ROLLING') != None

if __name__ == '__main__':
  device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
  model = build_model(fine_tune=False, weights=None).to(device)
  model.load_state_dict(torch.load("models/EfficientNet_1699276494_15.pt"))
  model.eval()
  foundation = get_foundation()
  outs = []
  for i, (frm, original_frame) in enumerate(gen_frames_from_file('data/test.mp4', foundation)):  
    out = model(torch.tensor(frm, dtype=torch.float32).to(device).unsqueeze(0)).cpu().item()
    outs.append(out)
    if ROLLING:
      out = np.mean(running_mean(np.array(outs[-38:]), WINDOWS))
    
    cv2.putText(original_frame, 'Predicted Speed: ' + str(round(out, 1)) + ' m/h', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (124,252,0), 2)
    cv2.imshow('frame', original_frame)
    if cv2.waitKey(1) == ord('q'):
      break