import os
import cv2
import numpy as np
import math
from sklearn import linear_model

color = np.random.randint(0, 255, (100, 3))

class VideoProcessor:
  def __init__(self, path):
    assert os.path.exists(path+'.mp4'), 'video path does not exist'
    self.video_capture = cv2.VideoCapture(path+'.mp4')
    self.lk_params = dict(winSize = (21, 21),
							      maxLevel = 2,
							      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    self.feature_params = dict(maxCorners = 30,
                      qualityLevel = 0.1,
                      minDistance = 10,
                      blockSize = 10)
    self.previous_points = None
    self.previous_gray_frame = None
    self.idx = 0
    self.speed_labels = open(path+'.txt', 'r')
  
  def construct_mask(self):
    H, W = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask = np.zeros(shape=(H, W), dtype=np.uint8)
    mask.fill(255)
    cv2.rectangle(mask, (0, 0), (W, H), (0, 0, 0), -1)

    x_top_offset = 240
    x_bottom_offset = 65

    poly_points = np.array([[[640-x_top_offset, 250], [x_top_offset, 250], [x_bottom_offset, 350], [640-x_bottom_offset, 350]]], dtype=np.int32)
    cv2.fillPoly(mask, poly_points, (255, 255, 255))
    self.mask = mask[130:350, 35:605]

  def process_frame(self, frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    self.previous_points = cv2.goodFeaturesToTrack(self.previous_gray_frame, mask=self.mask, **self.feature_params)
    
    try:
      if self.previous_points == None:
        return 0
    except:
      pass
  
    p1, st, err = cv2.calcOpticalFlowPyrLK(self.previous_gray_frame, frame, self.previous_points, None, **self.lk_params)
    flow = np.hstack((self.previous_points.reshape(-1, 2), (p1 - self.previous_points).reshape(-1, 2)))
    
    # Visualize
    if False:
		  # Select good points
      if p1 is not None:
          good_new = p1[st==1]
          good_old = self.previous_points[st==1]

      print(good_new, good_old)
      # draw the tracks
      for i, (new, old) in enumerate(zip(good_new, good_old)):
          a, b = new.ravel()
          c, d = old.ravel()
          mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
          frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
      img = cv2.add(frame, mask)
      cv2.imshow('frame', img)
      k = cv2.waitKey(30) & 0xff

    preds = []
    for x, y, u, v in flow:
      if v < -0.05:
        continue

      x -= frame.shape[1]/2
      y -= frame.shape[0]/2

      if y == 0 or (abs(u) - abs(v)) > 11:
        preds.append(0)
        preds.append(0)
      elif x == 0:
        preds.append(0)
        preds.append(v/(y*y))
      else:
        preds.append(u/(x*y))
        preds.append(v/(y*y))

    return np.array([n for n in preds if n >= 0])

if __name__ == "__main__":
  video_processor = VideoProcessor('data/train')
  video_processor.construct_mask()
  
  x_train = np.zeros(int(video_processor.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
  y_train = np.zeros(int(video_processor.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
  ret, frame = video_processor.video_capture.read()
  previous_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  previous_gray_frame = cv2.GaussianBlur(previous_gray_frame[130:350, 35:605], (3, 3), 0)

  i = 0
  while True:
    ret, frame = video_processor.video_capture.read()

    if not ret:
      break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = gray_frame[130:350, 35:605]

    video_processor.previous_gray_frame = previous_gray_frame
    previous_gray_frame = gray_frame
    frame = np.median(video_processor.process_frame(gray_frame))
    if math.isnan(frame): frame = 0
    x_train[i] = frame
    y_train[i] = np.float64(video_processor.speed_labels.readline())
    print(x_train[i], y_train[i])
    i+=1

  model = linear_model.LinearRegression(fit_intercept=False)
  model.fit(x_train.reshape(-1, 1), y_train) 
  hf_factor = model.coef_[0]
  print("Estimated hf factor = {}".format(hf_factor))

  pred_speed_train = x_train * hf_factor    
  mse = np.mean((pred_speed_train - y_train)**2)
  print(mse)



    


