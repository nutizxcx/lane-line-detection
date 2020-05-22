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

def rgb2gray(img):
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            new_img[x,y] = np.clip(round((int(img[x,y,0]) + int(img[x,y,1]) + int(img[x,y,2]))/3), 0, 255)
    return new_img

def rgb2hsv(img):
    hsv = np.zeros(img.shape, dtype = 'uint8')
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # change scale to 0 -> 1
            r = img[x,y,2]/255
            g = img[x,y,1]/255
            b = img[x,y,0]/255
            cmax = max(r,g,b)
            cmin = min(r,g,b)
            diff = cmax-cmin
            # V
            hsv[x,y,2] = cmax * 255
            # H
            if cmax == cmin:
                hsv[x,y,0] = 0
            elif cmax == r:
                hsv[x,y,0] = ((60 * (g-b)/diff) + 360) % 360
            elif cmax == g:
                hsv[x,y,0] = ((60 * (b-r)/diff) + 120) % 360
            elif cmax == b:
                hsv[x,y,0] = ((60 * (r-g)/diff) + 240) % 360
            hsv[x,y,0] = hsv[x,y,0]/2
            # S
            if cmax == 0:
                hsv[x,y,1] = 0
            else:
                hsv[x,y,1] = (diff / cmax) * 255
    return hsv

def bitwise_or(img1, img2):
    if ((img1.shape[0] != img2.shape[0]) and (img1.shape[1] != img2.shape[1])):
        print("image size error")
        return 0
    else:
        new_img = np.zeros(img1.shape, dtype = 'uint8')
        for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                if img1[x,y] or img2[x,y]:
                    new_img[x,y] = 255
                else:
                    new_img[x,y] = 0
        return new_img  

def bitwise_and(img1, img2):
    if ((img1.shape[0] != img2.shape[0]) and (img1.shape[1] != img2.shape[1])):
        print("image size error")
        return 0
    else:
        new_img = np.zeros(img1.shape, dtype = 'uint8')
        for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                if img1[x,y] and img2[x,y]:
                    # new_img[x,y] = round(((0.3*int(img1[x,y])) + (0.7*int(img2[x,y])))/2)
                    new_img[x,y] = 255
                else:
                    new_img[x,y] = 0
        return new_img            

def inRange(img, lower, upper):
    res = np.zeros((img.shape[0], img.shape[1]), dtype = 'uint8')
    for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if (img[x,y,0] >= lower[0] and img [x,y,0] <= upper[0]) and (img[x,y,1] >= lower[1] and img [x,y,1] <= upper[1]) and (img[x,y,2] >= lower[2] and img [x,y,2] <= upper[2]):
                    res[x,y] = 255
                else:
                    res[x,y] = 0
    return res

def main():
    # # read image
    img = cv2.imread('0376.jpg')

    # # Step 1
    g_img = rgb2gray(img)

    # # Step 2
    dg = darken_gray(g_img,0.7,0)

    # # Step 3
    hsv = rgb2hsv(img)

    # # Step 4 yellow mask
    lower_yellow = np.array([10,100,0], dtype=np.uint8)
    upper_yellow = np.array([40,255,255], dtype=np.uint8)
    yellow_mask = inRange(hsv, lower_yellow, upper_yellow)

    # # Step 5 white white
    lower_white = np.array([10,20,100], dtype=np.uint8)
    upper_white = np.array([30,50,255], dtype=np.uint8)
    white_mask = inRange(hsv, lower_white, upper_white)

    # # Step 6
    mask = bitwise_or(white_mask, yellow_mask)

    # # Step 7
    res = bitwise_and(dg,mask)

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