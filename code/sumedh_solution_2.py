from matplotlib import image
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
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # joining my masks
    my_mask = mask0 + mask1

    # removing all non-red regions
    bit_img = cv2.bitwise_and(image, image, mask = my_mask)
    row, col, channels = bit_img.shape
    print(row,col)
    bit_img = cv2.resize(bit_img,(int(row/5),int(col/5)))
    row, col, channels = bit_img.shape
    # red values of top and bottom pixels
    for i in range(row):
        for j in range(col):
            if bit_img[i, j, 0] != 0:
                count0 +=1
                if count0 == 1:
                    i = row - i
                    x_coordinate = np.append(x_coordinate, np.array([[i,j]]), axis = 0)

    for m in reversed(range(row)):
        for n in reversed(range(col)):
            if bit_img[m, n, 0] != 0:
                count1 += 1
                if count1 == 1:
                    m = row - m
                    y_coordinate = np.append(y_coordinate, np.array([[m,n]]), axis =  0)

    i = i + 1
    cv2.waitKey(1)

    return x_coordinate, y_coordinate

# Method to calculate least square                    
def leastSquareCalculator(input):
    x = input[:,0]
    y = input[:,1]
    x_x = np.power(x, 2)

    Mat = np.stack((x_x, x, np.ones((len(x)), dtype = int)), axis = 1)
    Mat_transpose = Mat.transpose()
    AtA = Mat_transpose.dot(Mat)
    AtY = Mat_transpose.dot(y)
    estimate = (np.linalg.inv(AtA)).dot(AtY)
    val = Mat.dot(estimate)

    return val

#Appending data values from each frame to an array
ball_vid_1 = cv2.VideoCapture('video1.mp4')
success_1, image_frame_1 = ball_vid_1.read()
c = 0
x_ = []
y_ = []

while success_1:

    x_vid, y_vid = ballCordinates(image_frame_1)
    for i in range(len(x_vid)):
        x_ = np.append(x_, ((x_vid[i][1]+y_vid[i][1])/2))
        
    for i in range(len(x_vid)):
	    y_ = np.append(y_,((x_vid[i][0]+y_vid[i][0])/2))
            
    success_1, image_frame_1 = ball_vid_1.read()
    c +=1

stack_ball_1 = np.vstack((x_, y_)).T

ball_vid_2 = cv2.VideoCapture('video2.mp4')
success_2, image = ball_vid_2.read()

c1 = 0
x_0 = []
y_0 = []
while success_2:

    x_vid, y_vid = ballCordinates(image)
    for i in range(len(x_vid)):
        x_0 = np.append(x_, ((x_vid[i][1]+y_vid[i][1])/2))
        
    for i in range(len(x_vid)):
	    y_0 = np.append(y_,((x_vid[i][0]+y_vid[i][0])/2))
            
    success_2, image = ball_vid_2.read()
    c1 +=1

stack_ball_2 = np.vstack((x_0, y_0)).T


leastSquare_1 = leastSquareCalculator(stack_ball_1)
leastSquare_2 = leastSquareCalculator(stack_ball_2)

fig = plt.figure()
plt.subplot(121)
plt.scatter(x_,y_,c='green', marker='^',label='Video 1 data points')
plt.plot(x_ ,leastSquare_1, c='red',linestyle='dashdot', label='Video 1 Least Squares')
plt.legend()
plt.subplot(122)
plt.scatter(x_0,y_0,c='pink', marker='^', label='Video 2 data points')
plt.plot(x_0 ,leastSquare_2, c='red', linestyle='dashdot', label='Video 2 Least Squares')
plt.legend()
plt.show()


ball_vid_1.release()
ball_vid_2.release()
cv2.destroyAllWindows()

