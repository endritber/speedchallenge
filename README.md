comma.ai programming speedchallenge!
======

Challenge
-----

Predict the speed of a car from video frames.

Data
-----

- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.

Training
----
```
# First, create segment of the training video:
./video.py

# Second, preprocess the data:
./preprocess.py

# Then, train the model:
./train.py

# To resume the training do:
LOAD=1 ./train.py <model_path>
```

Evaluation
-----

Mean squared error.
According to comma.ai an mse <10 is good. <5 is better. <3 is heart.