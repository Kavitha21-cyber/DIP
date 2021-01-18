# DIP
## 1.Develp program to display grayscale image using read and write operation.
## Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
## Importance of grayscaling –
## Dimension reduction:
 For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional. Reduces model complexity: Consider training neural  article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images. 
## program:
import numpy as np
import cv2
img = cv2.imread('goaL.jpg',0)
cv2.imshow('Original',img,)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('graygoal.jpg',img)
output
