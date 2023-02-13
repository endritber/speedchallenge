# comma.ai programming challenge!
-----
### Project
The goal of the challenge is to be able to predict the speed of a vehicle given a training video data footage from a dashcam. 

------
### Dataset
- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
-------

### Research
I had already thought that a convolution autoencoder and a [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) and a linear decoder would work on this challenge. However, GRU didn't work because because the model was reaching to a local minima and it was not learning.
As of the Machine Learning state of the art today I came across this [blog](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) where NVidia was trying to learn the system to steer with or without lane marking, I thought why not use it for velocity ego prediction. 
In this project I was also using dense optical flow computes the optical flow vector for every pixel of the frame which may be responsible for its slow speed but leading to a better accurate result.

### Data preprocessing
 - Brightness augmentation was utilized in order to fight overfitting. This prevented the network from simply looking for brightness characteristics and hopefully allowed the network to generalize to different conditions. This area seems understudied and could likely be taken further.
 - Resize 

