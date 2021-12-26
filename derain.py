import cv2
# from os import path
import os
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, draw_match, transform
from utils import (
    imshow, imread,
    write_and_show, destroyAllWindows,
    read_video_frames, write_frames_to_video
)


# read in video
video_name = 'image/rain2.mov'
images, fps = read_video_frames(video_name)


# get stabilized frames
stabilized = []
reference = images[0]
H, W = reference.shape[:2]
for img in tqdm(images[::2], 'processing'):
    # TODO: align all frames to reference frame (images[0])
    trans = ...



    stabilized.append(trans)
    imshow('trans.jpg', trans)

# write stabilized frames to a video
write_frames_to_video('stabilized.mp4', stabilized, fps/2)

# get rain free images
stabilized_mean = np.mean(stabilized, 0)
write_and_show('stabilized_mean.jpg', stabilized_mean)

destroyAllWindows()
