import cv2
# from os import path
import os
import numpy as np
from tqdm import tqdm
from sift import *
from extract_frame import read_video_frames

dir_name = 'image/night/'
filenames = os.listdir(dir_name)
filenames = [os.path.join(dir_name, f) for f in filenames]
images = [cv2.imread(f) for f in tqdm(filenames, 'read images')]
images = np.asarray(images)

# video_name = 'image/night.MOV'
# images = read_video_frames(video_name)
# images = np.asarray(images)

mean = images.mean(0)
cv2.imwrite('night_mean.jpg', mean)

stabilized = []
reference = images[0]
H, W = reference.shape[:2]
for img in tqdm(images, 'processing'):
    ref_kps, img_kps, match = keypoint_match(reference, img, max_n_match = 1000)
    # draw_match(reference, ref_kps, img, img_kps, match)
    img = np.float32(img)

    # get all matched keypoints
    ref_kps = np.array([ref_kps[m.queryIdx].pt for m in match])
    img_kps = np.array([img_kps[m.trainIdx].pt for m in match])

    trans = transform(img, img_kps, ref_kps, H, W)
    stabilized.append(trans)
stabilized = np.asarray(stabilized)

stabilized_mean = stabilized.mean(0)
cv2.imwrite('night_stabilized_mean.jpg', stabilized_mean)
