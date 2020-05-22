import cv2
import numpy as np
import sys

from s17 import *

#%%----------- Step 8,9 Canny edge detector ------------

def padding(image, padding_x, padding_y):
    # create padded image
    im_x, im_y = image.shape # find the size of image

    padded_image = np.zeros((im_x+2*padding_x, im_y+2*padding_y), dtype = int)
    padded_image[padding_x:-padding_x, padding_y:-padding_y] = image

    return padded_image

def convolution(image, kernel):
    # Convolution the image with certain odd-size kernel

    kernel_prime = np.flip(kernel) # flip 180 degrees for preparing convolution
    im_x, im_y = image.shape # find the size of image
    kern_x, kern_y = kernel.shape # find the size of kernel
    
    # create padded image
    conv_image = padding(image, kern_x//2, kern_y//2)

    conv_results = np.zeros((im_x, im_y)) # results declaration

    # convolution process
    for i in range(im_x):
        for j in range(im_y):
            conv_results[i,j] = np.sum(np.multiply(kernel_prime, conv_image[i:i+kern_x, j:j+kern_y]))

    return conv_results.astype('uint8')

def gaussianFilter(size, sd):
    # Using gaussian distribution to be a kernel by matrix = size x size

    assert size % 2 == 1, 'Size must be odd.'
    gaussian_filter = np.fromfunction(lambda x,y: 1/(2*np.pi*(sd**2))*np.exp(-((x-size//2)**2+(y-size//2)**2)/(2*sd**2)), (size,size)) # gaussian distribution with sd
    return gaussian_filter

def sobelEdgeDetection(image):
    # Using Sobel filter to detect edge

    sobel_x = np.array([[1,0,-1], [2,0,-2], [1,0,-1]]) # create vertical derivatives
    sobel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]]) # create horizontal derivatives
    
    # taking derivatives to the image
    image_x = convolution(image, sobel_x)
    image_y = convolution(image, sobel_y)

    # normalizing and thresholding
    magnitude = np.sqrt(image_x**2+image_y**2)
    normalized = magnitude * 255 / np.max(magnitude)

    # angle
    angle = np.arctan2(image_y, image_x) * 180/np.pi
    angle = np.where(angle > 0, angle // 22.5, 8 + angle // 22.5)
    normalized_angle = np.where(np.bitwise_or(angle == 0,angle == 7), 0,\
                        np.where(np.bitwise_or(angle == 1,angle == 2), 45,\
                        np.where(np.bitwise_or(angle == 3,angle == 4), 90, 135))) # Construct direction

    return normalized.astype('uint8'), normalized_angle.astype('uint8')

def nonMaximumSuppression(magnitude, angle):
    #edge thinning by using non-maximum suppression

    final = np.zeros(magnitude.shape,dtype = int)
    im_x, im_y = magnitude.shape # find the size of image
    padded = padding(magnitude, 1, 1) # create padded image

    # suppression method
    for i in range(im_x):
        for j in range(im_y):
            positive = 255; negative = 255
            if angle[i,j] == 0: positive = padded[i+2, j+1]; negative = padded[i, j+1] # when angle is 0, compare with east and west.
            elif angle[i,j] == 45: positive = padded[i+2, j+2]; negative = padded[i, j] # if 45, compare with NE and SW.
            elif angle[i,j] == 90: positive = padded[i+1, j+2]; negative = padded[i+1, j] # if 90, compare with N and S.
            else: positive = padded[i+2, j]; negative = padded[i, j+2] # if 135, compare with NW and SE.
            final[i,j] = magnitude[i, j] if max(positive, negative, magnitude[i, j]) == magnitude[i, j] else 0
    return final

def cannyEdgeDetection(image, gaussian_size, sd, lower_limit, upper_limit):
    # this function is applied for edge detection.

    gauss = gaussianFilter(gaussian_size, sd)
    convt_image = convolution(image, gauss)
    magnitude, angle = sobelEdgeDetection(convt_image)
    final = nonMaximumSuppression(magnitude, angle)
    final[np.bitwise_or(lower_limit > final, final > upper_limit)] = 0
    final[final != 0] = 255

    return final.astype('uint8')
#%%------------ Step 10 Define region of interest ----------------------