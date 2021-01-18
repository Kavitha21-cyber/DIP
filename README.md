# DIP
## 1.Develp program to display grayscale image using read and write operation.
## Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
## Importance of grayscaling –
## Dimension reduction:
 For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional. Reduces model complexity: Consider training neural  article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images. 
## program:
## imshow()
This function in pyplot madules of matplotlib lib is used display as an image.
## imwrite()
This method is used to save an image to any storage devices.
## waitKey()
It is a keyboard binding function, the function waits for specified miliseconds for any keyboard event .If you press any key in that time,the program continues.
## destroyAllWindows
simply destroys all the windows we created.

import numpy as np
import cv2
img = cv2.imread('goaL.jpg',0)
cv2.imshow('Original',img,)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('graygoal.jpg',img)
output
![image](https://user-images.githubusercontent.com/72538198/104898756-05b01f80-59a0-11eb-9da6-66e57376a634.png)
## 2.Develop a program to perform linear transformation on image.
## Scalar transformation:
Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image .
## cv2.resize()
method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image
## imshow()
function in pyplot module of matplotlib library is used to display data as an image
import cv2
import numpy as np
src=cv2.imread('goal.jpg',1)
img=cv2.imshow('goal.jpg',src)
scale_p=500
width=int(src.shape[1]*scale_p/100)
height=int(src.shape[0]*scale_p/100)
dsize=(width,height)
result=cv2.resize(src,dsize)
cv2.imwrite('scaling.jpg',result)
cv2.waitKey(0)
Output:
