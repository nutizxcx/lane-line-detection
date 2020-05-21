import cv2 as cv
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def gaussianPyramid(img, reduceScale):
    original_img = img.copy()
    newImg = []
    tempRow = []
    for i in range(reduceScale):
        img = gaussian_filter(img, sigma=1)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if (col % 2 == 0) and (row % 2 == 0):
                    tempRow.append(img[row,col])                    
            if (row % 2 == 0):
                newImg.append(tempRow)
                tempRow = []
        img = newImg.copy()      
        newImg = []
    print("img size:", original_img.shape[:2])
    print("new img size:", np.array(img).shape[:2])
    return np.array(img)
            
def houghline(img, imgForDraw, threshold, reduceScale):
    width, height = img.shape[:2]
    diagonal = math.floor(np.sqrt(width**2 + height**2))
    lines = np.zeros((2*diagonal,180))
    possible_rho = np.array(range(-diagonal,diagonal))

    t0 = time.time()
    for col in range(height):
        for row in range(width):
            # if pixel is not background
            if img[row,col] > 0:
                # calculate rho,theta coefficient
                for theta in range(lines.shape[1]):
                    # scale theta and convert theta to radian
                    rad_theta = (theta - 90) * math.pi / 180
                    # calculate distance 
                    dist = (col * math.cos(rad_theta) + row * math.sin(rad_theta))
                    # calculate diff between calculated distance and possible rho
                    d = min(abs(possible_rho - dist))
                    # acceptable diff range <= 1 (because we cannot check all possible rho values)
                    if d <= 1:
                        index = math.floor(dist + lines.shape[0]/2)
                        lines[index,theta] = lines[index,theta] + 1 
    t1 = time.time()
    print("total process time: %.2f" % float(t1-t0),"sec")

    # draw lines
    rho_theta_coor = np.where(lines>threshold)
    for rho, theta in zip(rho_theta_coor[0],rho_theta_coor[1]):
        rho = rho - diagonal
        rho = rho * ( 2 ** reduceScale ) 
        rad_theta = (theta - 90)*math.pi / 180
        a = np.cos(rad_theta)
        b = np.sin(rad_theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1500*(-b))
        y1 = int(y0 + 1500*(a))
        x2 = int(x0 - 1500*(-b))
        y2 = int(y0 - 1500*(a))
        cv.line(imgForDraw,(x1,y1),(x2,y2),(0,255,0),1)
    return [lines,imgForDraw]

