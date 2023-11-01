#!/usr/bin/env python3
from video import  get_foundation, gen_frames_from_file
import os
import tqdm
import numpy as np
import torch

DATA_PATH = 'data/segments'
OUTPUT_PATH = 'data/preprocessed'

def preprocess(foundation, filename, filename_out, transform=None):
  if not filename.endswith(".mp4"): return
  if os.path.isfile(filename_out): return
  if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
  print('preprocessing', filename, filename_out)
  frms = []
  for frm in tqdm.tqdm(gen_frames_from_file(filename, foundation)):
    frms.append(frm)

  big_x = torch.Tensor((np.concatenate([x.reshape(1, 160, 320, 3) for x in frms])))
  print(f"Saving -> shape:{big_x.shape}")
  torch.save({"x": big_x}, filename_out)

if __name__ == '__main__':
  foundation = get_foundation()
  for filename in os.listdir(DATA_PATH):
    if filename.endswith('.mp4'):
      preprocess(foundation, os.path.join(DATA_PATH, filename), os.path.join(OUTPUT_PATH, filename.replace('.mp4', ".pt")), None)