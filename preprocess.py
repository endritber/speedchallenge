#!/usr/bin/env python3
from video import  get_foundation, gen_frames_from_file
import os
import tqdm
import numpy as np
import torch

DATA_PATH = 'data/splits'
OUTPUT_PATH = 'data/preprocessed'

def preprocess(foundation, filename, filename_out, transform=None):
  if not filename.endswith(".mp4"): return
  if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
  print('preprocessing', filename, filename_out)
  if os.path.isfile(filename_out): return
  frms = []
  for frm in tqdm.tqdm(gen_frames_from_file(filename, foundation)):
    frms.append(frm)

  if frms:
    big_x = torch.from_numpy(np.array(frms).astype(np.float32))
    print('Saving:', big_x.shape, big_x.dtype)
    torch.save(big_x, filename_out)

if __name__ == '__main__':
  foundation = get_foundation()
  for filename in os.listdir("data/splits/"):
    if filename.endswith('.mp4'):
      preprocess(foundation, os.path.join("data/splits/", filename), os.path.join(OUTPUT_PATH, filename.replace('.mp4', ".pt")), None)