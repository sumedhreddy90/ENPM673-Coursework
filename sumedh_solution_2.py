import numpy as np
import cv2
import math
import matplotlib
from sympy import N
import matplotlib.pyplot as plt

# Finding center of the ball 
# by splitting red from the image frame

def ballCordinates(image):
    x_coordinate = np.empty((0,2), int)
    y_coordinate = np.empty((0,2), int)
    count0 = 0 
    count1 = 0

    # Convert BGR to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # joining my masks
    my_mask = mask0 + mask1

    # removing all non-red regions
    bit_img = cv2.bitwise_and(image, image, my_mask)
    row, col, channels = bit_img.shape
    bit_img = cv2.resize(bit_img,(int(row/5)),(int(col/5)))
    row, col, channels = bit_img.shape

    # red values of top and bottom pixels
    for i in range(row):
        for j in range(col):
            if bit_img[i, j, 0] != 0:
                count0 +=1
                if count0 == 1:
                    i = row - i
                    x_coordinate = np.append(x_coordinate, np.array([[i,j]]), axis = 0)

    for l in reversed(range(row)):
        for k in reversed(range(col)):
            if bit_img[l, k, 0] != 0:
                count1 += 1
                if count1 == 1:
                    l = row - l
                    y_coordinate = np.append(y_coordinate, np.array([[l,k]]), axis =  0)

    i = i + 1
    cv2.waitKey(1)
    return x_coordinate, y_coordinate
                    
def leastSquareCalculator():
    

