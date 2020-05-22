import cv2 
import matplotlib.pyplot as plt
import numpy as np
from s17copy import *
from edgedetection import *
from houghline import *


t0 = time.time()
img = cv2.imread('dataset/0376.jpg')
imgForDraw = img.copy()
temp = np.zeros((img.shape[0], img.shape[1]))

# Region of interest
img = img[img.shape[0]-150:img.shape[0]-30, :img.shape[1]]

# Step 1
g_img = rgb2gray(img)

# Step 2
dg = darken_gray(g_img,0.7,0)

# Step 3
hsv = rgb2hsv(img)

# Step 4 yellow mask
lower_yellow = np.array([0,68,108], dtype=np.uint8)
upper_yellow = np.array([23,255,255], dtype=np.uint8)
yellow_mask = inRange(hsv, lower_yellow, upper_yellow)

# Step 5 white white
lower_white = np.array([23,0,103], dtype=np.uint8)
upper_white = np.array([43,25,169], dtype=np.uint8)
white_mask = inRange(hsv, lower_white, upper_white)


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
# white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

# Step 6
mask = bitwise_or(white_mask, yellow_mask)

# Step 7
res = bitwise_and(dg,mask)

temp[temp.shape[0]-150:temp.shape[0]-30, :temp.shape[1]] = bitwise_or(res, temp[temp.shape[0]-150:temp.shape[0]-30, :temp.shape[1]])
res = temp.copy()
# cv2.imshow('res',res)
# cv2.waitKey()

# try:
edge = cannyEdgeDetection(res, 7, 2, 250, 255)
cv2.imshow('edge',edge)
cv2.waitKey()

# except:
#     print("Unexpected error:\n", sys.exc_info()[0], '\n', sys.exc_info()[1], 'at line', str(sys.exc_info()[2].tb_lineno)+'.')
#     input('')
#     raise
# else:
#     input("\nNormal Termination. Goodbye!")

# img = edge.copy()
# pyramidLayer = 3
# img = gaussianPyramid(img,pyramidLayer)
# original_img = gaussianPyramid(imgForDraw,pyramidLayer)
# # img = cv2.ximgproc.thinning(img)


# # for i in range(img.shape[0]):
# #     for j in range(img.shape[1]):
# #         if img[i,j] < 10:
# #             img[i,j] = 0
# #         else:
# #             img[i,j] = 255

# thres = 35
# [houghLineRep,result] = houghline(img, imgForDraw, thres, pyramidLayer)

# t1 = time.time()
# print("total process time: %.2f" % float(t1-t0),"sec")

# while True:
#     cv2.imshow("original_img",original_img)
#     cv2.imshow("houghLineRep",houghLineRep)
#     cv2.imshow("result",result)
#     if cv2.waitKey() & 27:
#         break



# # fig=plt.figure(figsize=(8, 8))
# # columns = 3
# # rows = 1
# # fig.add_subplot(rows, columns, 1)
# # plt.imshow(original_img)
# # fig.add_subplot(rows, columns, 2)
# # plt.imshow(houghLineRep, cmap='gray')
# # fig.add_subplot(rows, columns, 3)
# # plt.imshow(result)
# # plt.show()
















# # img = cv.imread("test image/test5.jpg")
# # imgForDraw = img.copy()
# # pyramidLayer = 3
# # original_img = gaussianPyramid(img,pyramidLayer)

# # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
# # # img = cv.Canny(img,100,100,apertureSize = 3)
# # img = cv.ximgproc.thinning(img)
# # img = gaussianPyramid(img,pyramidLayer)

# # # implement global thresholding
# # for i in range(img.shape[0]):
# #     for j in range(img.shape[1]):
# #         if img[i,j] < 10:
# #             img[i,j] = 0
# #         else:
# #             img[i,j] = 255

# # thres = 18
# # [houghLineRep,result] = houghline(img, imgForDraw, thres, pyramidLayer)

# # # show image
# # fig=plt.figure(figsize=(8, 8))
# # columns = 3
# # rows = 1
# # fig.add_subplot(rows, columns, 1)
# # plt.imshow(original_img)
# # fig.add_subplot(rows, columns, 2)
# # plt.imshow(houghLineRep, cmap='gray')
# # fig.add_subplot(rows, columns, 3)
# # plt.imshow(result)
# # plt.show()