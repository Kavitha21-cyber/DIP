# DIP
1.Develp program to display grayscale image using read and write operation.
import numpy as np
import cv2
img = cv2.imread('goaL.jpg',0)
cv2.imshow('Original',img,)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('graygoal.jpg',img)
