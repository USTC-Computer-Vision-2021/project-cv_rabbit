import numpy as np
from utils import imread, write_and_show, destroyAllWindows
from sift import keypoint_match, draw_match, transform

if __name__ == '__main__':
    
    ## read images
    img1 = imread('../image/now.jpg')
    img2 = imread('../image/past_gray.jpg')

    ## find keypoints and matches
    keypoints1, keypoints2, match = keypoint_match(img1, img2, max_n_match=1000)

    draw_match(img1, keypoints1, img2, keypoints2,
               match, savename='match.jpg')

    # get all matched keypoints
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])

    ## Align img2 to img1
    H, W = img1.shape[:2]
    new_img2 = transform(img2, keypoints2, keypoints1, H, W)
    write_and_show('past_transformed.jpg', new_img2)



    # resize img1
    #new_img1 = np.hstack([img1, np.zeros_like(img1)])
    index = new_img2!=0
    img1[index] = 0
    new_img1 = img1
    write_and_show('now_transformed.jpg', new_img1)

    # TODO: average `new_img1` and `new_img2`
    cnt = np.zeros([H,W,1]) + 1e-10
    cnt += (new_img2 != 0).any(2, keepdims=True)
    cnt += (new_img1 != 0).any(2, keepdims=True)
    new_img1 = np.float32(new_img1)
    new_img2 = np.float32(new_img2)
    stack = (new_img2+new_img1)/cnt

    write_and_show('stack.jpg', stack)

    destroyAllWindows()
