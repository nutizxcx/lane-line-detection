import cv2
import numpy as np
import colorsys

def darken_gray(g_img,alpha,beta):
    # defualt alpha = 1.0, beta = 0
    new_img = np.zeros(g_img.shape, g_img.dtype)
    for x in range(g_img.shape[0]):
        for y in range(g_img.shape[1]):
            new_img[x,y] = np.clip(alpha * g_img[x,y] - beta, 0, 255) 
    return new_img

def main():
    # read image
    img = cv2.imread('dataset/0376.jpg')

    # Step 1
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2
    dg = darken_gray(g_img,0.7,0)

    # Step 3
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 4 yellow mask
    lower_yellow = np.array([10,100,0], dtype=np.uint8)
    upper_yellow = np.array([40,255,255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Step 5 white white
    lower_white = np.array([10,20,100], dtype=np.uint8)
    upper_white = np.array([30,50,255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Step 6
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Step 7
    res = cv2.bitwise_and(dg,mask)

    cv2.imshow('yellow',yellow_mask)
    cv2.imshow('white',white_mask)
    cv2.imshow('image',img)
    cv2.imshow('darken',dg)
    cv2.imshow('hsv',hsv)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.waitKey()

if __name__ == "__main__":
    main()