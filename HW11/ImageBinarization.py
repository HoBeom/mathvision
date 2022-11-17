import cv2
import numpy as np
# code from https://github.com/SeolMuah/mathvision/blob/master/HW11_Least_Square/q1_main.py
# 1. Obtain the best binarized image of the provided sample image by a global thresholding (not by adaptive thresholding)
img = cv2.imread('hw11_sample.png', cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('THRESH_BINARY', thresh)
cv2.waitKey(0)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('THRESH_OTSU', thresh)
cv2.waitKey(0)


# 2. Approximate the background of the sample image by a 2nd order polynomial surface and then display it as an image
xs = np.arange(0, img.shape[1])
ys = np.arange(0, img.shape[0])
x, y = np.meshgrid(xs, ys)
pos = np.dstack((x, y))
pos = pos.reshape(-1, 2)
A = []
I = []
mean_x = pos[:, 0].mean()
std_x = pos[:, 0].std()
mean_y = pos[:, 1].mean()
std_y = pos[:, 1].std()
norm_x = (pos[:, 0] - mean_x) / std_x
norm_y = (pos[:, 1] - mean_y) / std_y
A = np.vstack((norm_x, norm_y, norm_x * norm_y, norm_x ** 2, norm_y ** 2, np.ones_like(norm_x))).T
I = np.array(img.reshape(-1, 1))
A_plus = np.linalg.inv((A.T @ A)) @ A.T
P = A_plus @ I
background = A @ P
background = background.reshape(img.shape)
cv2.imshow('background', background.astype(np.uint8))
cv2.waitKey(0)

# 3. Subtract the approximated background image from the original and binarize the result (background-subtracted image) to obtain the final best binarized
subtracted = img.astype(np.float32) - background.astype(np.float32)
subtracted += subtracted.min()
subtracted = subtracted.astype(np.uint8)
cv2.imshow('subtracted', subtracted)
cv2.waitKey(0)
ret, thresh = cv2.threshold(subtracted, -1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('subtracted THRESH_OTSU', thresh)
cv2.waitKey(0)



