from houghline import gaussianPyramid, houghline
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("test image/test5.jpg")
imgForDraw = img.copy()
pyramidLayer = 3
original_img = gaussianPyramid(img,pyramidLayer)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
img = cv.Canny(img,50,100,apertureSize = 3)
img = cv.ximgproc.thinning(img)
img = gaussianPyramid(img,pyramidLayer)

# implement global thresholding
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j] < 10:
            img[i,j] = 0
        else:
            img[i,j] = 255

thres = 18
[houghLineRep,result] = houghline(img, imgForDraw, thres, pyramidLayer)

# show image
fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 1
fig.add_subplot(rows, columns, 1)
plt.imshow(original_img)
fig.add_subplot(rows, columns, 2)
plt.imshow(houghLineRep, cmap='gray')
fig.add_subplot(rows, columns, 3)
plt.imshow(result)
plt.show()