import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from sympy import N


# Square shaped camera sensor properties 
cam_width= 14 
cam_height= 14
focal_len = 25

# Calculating horizontal and vertical FOV
width_half_angle = math.atan2(cam_width,(2*focal_len))
height_half_angle = math.atan2(cam_height,(2*focal_len))
vertical_FOV = 2 * width_half_angle
horizontal_FOV = 2 * height_half_angle

print("Horizontal FOV ::: ", N(horizontal_FOV,2))
print("Vertical FOV ::: ", N(vertical_FOV,2))

#Calculating minimum number of pixels of the object in the image
obj_width= 5 
obj_height= 5
cam_dist =20 
resol = 5000000 
pixel_w = math.sqrt(resol)
pixel_h = math.sqrt(resol)

OP_height = (focal_len * obj_height * pixel_h)/(cam_dist * cam_height)
Minimum_OP = OP_height * OP_height

print("The min no of pixels occupied by the object in the image is ", round(Minimum_OP) )