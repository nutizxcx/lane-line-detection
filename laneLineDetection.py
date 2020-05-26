import cv2 
import matplotlib.pyplot as plt
import numpy as np
from colorAdjustment import *
from edgedetection import *
from houghline import *


t0 = time.time() 
img = cv2.imread('dataset/0376.jpg')
originalImg = img.copy()
imgForDraw = img.copy()
temp = np.zeros((img.shape[0], img.shape[1]))

# Step 1
g_img = rgb2gray(img)

# Step 2
dg = darken_gray(g_img,0.7,0)

# Region of interest
img = img[img.shape[0]-150:img.shape[0]-30, :img.shape[1]]

# Step 3
hsv = rgb2hsv(img)

# Step 4 yellow mask
lower_yellow = np.array([0,68,108], dtype=np.uint8)
upper_yellow = np.array([23,255,255], dtype=np.uint8)
yellow_mask = inRange(hsv, lower_yellow, upper_yellow)

# Step 5 white white
lower_white = np.array([24,0,103], dtype=np.uint8)
upper_white = np.array([62,255,255], dtype=np.uint8)
white_mask = inRange(hsv, lower_white, upper_white)

# Step 6
mask = bitwise_or(white_mask, yellow_mask)

# Step 7
res = bitwise_and(dg[dg.shape[0]-150:dg.shape[0]-30, :dg.shape[1]], mask)

temp[temp.shape[0]-150:temp.shape[0]-30, :temp.shape[1]] = res

res = temp.copy()

# Edge detection
edge = cannyEdgeDetection(res, 3, 5, 50, 100)

img = edge.copy()
# Apply Gaussian pyramid
pyramidLayer = 3
img = gaussianPyramid(img,pyramidLayer)
# vote for interest line
thres = 20
[houghLineRep,result] = houghline(img, imgForDraw, thres, pyramidLayer)

t1 = time.time()
print("total process time: %.2f" % float(t1-t0),"sec")

# show result
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 1
fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB))
fig.add_subplot(rows, columns, 2)
plt.imshow(res, cmap='gray')
fig.add_subplot(rows, columns, 3)
plt.imshow(edge, cmap='gray')
fig.add_subplot(rows, columns, 4)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# fig.add_subplot(rows, columns, 5)
# plt.imshow(houghLineRep, cmap='gray')
plt.show()











