import imageio
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, transform
from utils import read_video_frames, write_and_show, destroyAllWindows, imshow

video_name = '../image/Vcore.mov'
images, fps = read_video_frames(video_name)
n_image = len(images)

images_out = [images[i] for i in range(n_image) if i%10==0]
imageio.mimsave('../result/Vcore.gif', images_out, 'GIF', duration=0.1)

# TODO: init panorama
h, w = images[0].shape[:2]
H, W = h*4, w*3
panorama = np.zeros([H,W,3]) # use a large canvas

h_start = H-h-h//2
w_start = W-w
panorama[h_start:h_start+h, w_start:w_start+w, :] = images[0]
imshow('panorama.jpg', panorama)
#print(panorama[h_start:h_start+h, w_start:w_start+w, :])
trans_sum = np.zeros([H,W,3])
cnt = np.ones([H,W,1])*1e-10

for img in tqdm(images[::4], 'processing'):
    # TODO: stitch img to panorama one by one
    keypoints1, keypoints2, match = keypoint_match(panorama, img, max_n_match=1000, draw=False)
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])
    aligned_img = transform(img, keypoints2, keypoints1, H, W)
    
    trans_sum += aligned_img
    cnt += (aligned_img != 0).any(2, keepdims=True)
    panorama = trans_sum/cnt
    # show
    imshow('panorama.jpg', panorama)


# panorama = algined.mean(0)
write_and_show('panorama.jpg', panorama)

destroyAllWindows()
