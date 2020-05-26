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
    return np.array(img)

def findStrongLine(rho_theta_coor):
    strongLine = []
    distLim = 140
    maxLine = 4

    for i in range(len(rho_theta_coor[0])):
        for j in range(i,len(rho_theta_coor[1])):
            dist = math.sqrt( ( (rho_theta_coor[0][i] - rho_theta_coor[0][j])**2 ) + (rho_theta_coor[1][i] - rho_theta_coor[1][j])**2 ) 
            if dist > distLim:          
                strongLine.append([rho_theta_coor[0][i], rho_theta_coor[1][i]])
                strongLine.append([rho_theta_coor[0][j], rho_theta_coor[1][j]])
                break
        strongLine = [x for n,x in enumerate(strongLine) if x not in strongLine[:n]]
        if len(strongLine) == maxLine:
            break

    if len(strongLine) == 0:
        strongLine = [[rho_theta_coor[0][i], rho_theta_coor[1][i]]]

    return strongLine

def drawLine(imgForDraw, lines, threshold, diagonal, reduceScale, startPoint, endPoint):
    # draw lines
    rho_theta_coor = np.where(lines>threshold)
    if len(rho_theta_coor[0]) != 0:
        strongLine = findStrongLine(rho_theta_coor)
        for rho, theta in strongLine:
            x1 = startPoint[rho][theta][1] * ( 2 ** reduceScale ) 
            y1 = startPoint[rho][theta][0] * ( 2 ** reduceScale )
            x2 = endPoint[rho][theta][1] * ( 2 ** reduceScale )
            y2 = endPoint[rho][theta][0] * ( 2 ** reduceScale )
            cv.line(imgForDraw,(x1,y1),(x2,y2),(0,255,0),3)
    return [lines,imgForDraw]

def houghline(img, imgForDraw, threshold, reduceScale):
    height, width = img.shape[:2]
    diagonal = math.floor(np.sqrt(width**2 + height**2))
    lines = np.zeros((2*diagonal,180))
    # create empty startPoint list
    startPoint = []
    for i in range(lines.shape[0]):
        tmp = [ [] for _ in range(lines.shape[1])]
        startPoint.append(tmp)
    # create empty endPoint list
    endPoint = []
    for i in range(lines.shape[0]):
        tmp = [ [] for _ in range(lines.shape[1])]
        endPoint.append(tmp)

    possible_rho = np.array(range(-diagonal,diagonal))

    for row in range(height):
        for col in range(width):
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
                        # initial start point                    
                        if len(startPoint[index][theta]) == 0:
                            startPoint[index][theta] = [row,col]
                        # initial end point
                        elif len(endPoint[index][theta]) == 0:
                            endPoint[index][theta] = [row,col]
                        # find longest end point
                        else:
                            previousDist = math.sqrt( (startPoint[index][theta][0] - endPoint[index][theta][0])**2 + (startPoint[index][theta][1] - endPoint[index][theta][1])**2 )
                            newDist = math.sqrt( (startPoint[index][theta][0] - row)**2 + (startPoint[index][theta][1] - col)**2 )
                            if previousDist < newDist:
                                endPoint[index][theta] = [row,col]    
    return drawLine(imgForDraw, lines, threshold, diagonal, reduceScale, startPoint, endPoint)

