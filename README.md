# 基于SIFT实现图像拼接（A look into the past）
核心技术要点：
1. 关键点检测（Keypoints Detection）
2. 匹配（Match）
3. 拼接（Image Stitich）

成员及分工：
1. 王志强 PB18051049
    - 算法设计
    - 代码实现
    - 文档撰写
2. 蒲明昱 PB18111733
    - 方案调研
    - 算法设计
    - 文档撰写

## 问题描述
1. 初衷和动机：[A look into the past](https://blog.flickr.net/en/2010/01/27/a-look-into-the-past)是一种图片艺术，让照片有了“昨日重现”的效果，很适合发朋友圈。所以，我们利用Computer Vision课堂上学到的知识，在一些我们感兴趣的照片上实现“A look into the past”。
2. 创意描述：打开手机相册，找一些有意思的照片，对某个部位进行截取。当然，如果会用PS的话，可以对该部分做更精细化的截取甚至做一些特效，可以得到更好的效果。因为懒，我们就直接截取了图片某个部分，将其转化为灰度图像，之后通过计算机视觉算法对两张图进行处理，最终得到“A look into the past”的效果。
3. 计算机视觉问题：上述创意的实现，可以转化为计算机视觉中的图像拼接问题，关键技术是关键点检测、匹配以及图像的缝合。

## 原理分析

OpenCV(开源计算机视觉库)是一个功能强大的计算机视觉库。这个库是跨平台的，在开源的Apache 2许可下可以免费使用。OpenCV是用c++编写的，它的主要接口是用c++编写的，但它也提供了一个Python接口。我们依靠它来检测关键点、匹配关键点和转换图像。

###  关键点检测（Keypoints Detection）
#### 关键点检测的简要历史
OpenCV-Python^[[Feature Detection and Description -- OpenCV-Python Tutorials beta documentation](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)]教程对这段简短的历史和回顾进行了总结。

1. Harris Corner Detection:^[Harris, C., & Stephens, M. (1988, August). A combined corner and edge detector. In Alvey vision conference (Vol. 15, No. 50, pp. 10-5244).]

    One early attempt to find these corners was done by Chris Harris & Mike Stephens in their paper A Combined Corner and Edge Detector in 1988, so now it is called Harris Corner Detector.

2. SIFT (Scale-Invariant Feature Transform):^[Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.]

    Harris corner detector is not good enough when scale of image changes. D.Lowe developed a breakthrough method to find scale-invariant features and it is called SIFT in 2004.

3. SURF (Speeded-Up Robust Features):^[Bay, H., Tuytelaars, T., & Van Gool, L. (2006, May). Surf: Speeded up robust features. In European conference on computer vision (pp. 404-417). Springer, Berlin, Heidelberg.]

    SIFT is really good, but not fast enough. In 2006, three people, Bay, H., Tuytelaars, T. and Van Gool, L, introduced a new algorithm called SURF. As name suggests, it is a speeded-up version of SIFT.

4. ORB (Oriented FAST and Rotated BRIEF):^[Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, November). ORB: An efficient alternative to SIFT or SURF. In 2011 International conference on computer vision (pp. 2564-2571). IEEE.]

    This algorithm was brought up by Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary R. Bradski in their paper ORB: An efficient alternative to SIFT or SURF in 2011. As the title says, it is a good alternative to SIFT and SURF in computation cost, matching performance and mainly the patents. Yes, SIFT and SURF are patented and you are supposed to pay them for its use. But ORB is not !!! (correction: the patent of SIFT has already expired now).

In summary, Harris is the early one containing basic idea of corner detection. SIFT is the first mature one, but slow. SURF is a speeded-up version of SIFT. ORB is a free alternative for SIFT and SURF.

However, the patent of SIFT expired in March of 2020, so SIFT is free to use now ✌️! But patent of SURF is still valid now 🙁.