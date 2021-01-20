## DIP
## 1.Develp program to display grayscale image using read and write operation.
## Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
## Importance of grayscaling –
## Dimension reduction:
 For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional. Reduces model complexity: Consider training neural  article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images. 
## imshow()
This function in pyplot madules of matplotlib lib is used display as an image.
## imwrite()
This method is used to save an image to any storage devices.
## waitKey()
It is a keyboard binding function, the function waits for specified miliseconds for any keyboard event .If you press any key in that time,the program continues.
## destroyAllWindows
simply destroys all the windows we created.
## PROGRAM:
import numpy as np
import cv2
img = cv2.imread('goaL.jpg',0)
cv2.imshow('Original',img,)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('graygoal.jpg',img)
## OUTPUT:
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
## OUTPUT:
![image](https://user-images.githubusercontent.com/72538198/104899706-304ea800-59a1-11eb-83cb-0d56cce9298f.png)
## ROTATION
Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal, flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. OpenCV is a well-known library used for image processing.
## PROGRAM
import cv2
import numpy as np
src=cv2.imread('goal.jpg')
img=cv2.imshow('goal.jpg',src)
windowsname='image'
image=cv2.rotate(src,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow(windowsname,image)
cv2.waitKey(0)
## OUTPUT
![image](https://user-images.githubusercontent.com/72538198/104900446-1497d180-59a2-11eb-8004-7554fc6b1ed6.png)

## 3.Develop a program to find sum and mean of multiple images.
You can add two images with the OpenCV function, cv. add(), or simply by the numpy operation res = img1 + img2. The function mean calculates the mean value M of array elements, independently for each channel, and return it:" This mean it should return you a scalar for each layer of you image
## The append()
method in python adds a single item to the existing list.
## listdir() 
method in python is used to get the list of all files and directories in the specified directory.

## program:
import cv2 
import os
path='C:\pyth'
img=[]
files=os.listdir(path)
for file in files:
    fpath=path+"\\"+file
    img.append(cv2.imread(fpath))
i=0
im=[]
for im in img:
    im+=img[i]
    i=i+1
cv2.imshow("sum",im)
mean=im/len(files)
cv2.imshow("mean",mean)
cv2.waitKey(0)
## OUTPUT:
![image](https://user-images.githubusercontent.com/72538198/104902314-9ab51780-59a4-11eb-860d-c8c67f996809.png)
## 4.Convert images grayscale and binary image.
## Grayscaling 
 it is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
## A binary image 
 it is a monochromatic image that consists of pixels that can have one of exactly two colors, usually black and white.
## cv2.threshold
 it works as, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black).
## program:
import numpy as np
import cv2
img = cv2.imread('goal.jpg',0)
cv2.imshow('Original',img)
cv2.imwrite('graygoal.jpg',img)
img = cv2.imread('goal.jpg', 2) 
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
cv2.imshow("Binary", bw_img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
## OUTPUT:
![image](https://user-images.githubusercontent.com/72538198/104902512-d6e87800-59a4-11eb-98fd-530ed20b6156.png)
## 5.Develop program to given colour images to different colour space.
Color spaces are a way to represent the color channels present in the image that gives the image that particular hue
## BGR color space: 
OpenCV’s default color space is RGB.
## HSV color space:
It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye. Hue value varies from 0-179, Saturation value varies from 0-255 and Value value varies from 0-255.
## LAB color space :
L – Represents Lightness.A – Color component ranging from Green to Magenta.B – Color component ranging from Blue to Yellow. The HSL color space, also called HLS or HSI, stands ## for:Hue : the color type Ranges from 0 to 360° in most applications
## Saturation :
variation of the color depending on the lightness.
## Lightness :
(also Luminance or Luminosity or Intensity). Ranges from 0 to 100% (from black to white).
## YUV:
Y refers to the luminance or intensity, and U/V channels represent color information. This works well in many applications because the human visual system perceives intensity information very differently from color information.
## cv2.cvtColor()
method is used to convert an image from one color space to another.

## program:
import cv2
img=cv2.imread('goal.jpg')
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('yuv image', yuv_img)
cv2.waitKey()
cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey()
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV image', hsv_img)
cv2.waitKey()
cv2.imshow('h channel',hsv_img[:,:,0])
cv2.imshow('s channel',hsv_img[:,:,1])
cv2.imshow('v channel',hsv_img[:,:,2])
cv2.waitKey()
## OUTPUT: 
![image](https://user-images.githubusercontent.com/72538198/104905069-e2896e00-59a7-11eb-98c0-c4cc6b715d40.png)
## 6.Develop a program to create an images from 2D array.Generate array of random size.
## 2D array can be defined as an array of arrays. The 2D array is organized as matrices which can be represented as the collection of rows and columns. However, 2D arrays are created to implement a relational database look alike data structure.
## numpy.zeros()
function returns a new array of given shape and type, with zeros.
## Image.fromarray(array)
is creating image object of above array

## program:
import numpy as np
from PIL import Image
array = np.linspace(0,1,200*220)
mat = np.reshape(array,(200,220))
img = Image.fromarray( mat , 'RGB')
img.show()
## OUTPUT:
![image](https://user-images.githubusercontent.com/72538198/104905276-22e8ec00-59a8-11eb-81a5-7c1ba8fa5abd.png)

## 7.find the neighborhood values of the matrix
## SUM OF NEIGHBORS :
import numpy as np
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself
M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 
M = np.asarray(M)
N = np.zeros(M.shape)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print("Original matrix:\n",M)
print("Summed neighbors matrix:\n",N)

![image](https://user-images.githubusercontent.com/72538198/104907323-ed91cd80-59aa-11eb-9f8c-24b67c2c0dbc.png)
## program 8
finding the neighbours of the matrix:
import numpy as np
i=0
j=0
a= np.array([[1,2,3,4,5], [2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
print("a : ",str(a))
def neighbors(radius, rowNumber, columnNumber):
     return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
                for j in range(columnNumber-1-radius, columnNumber+radius)]
                    for i in range(rowNumber-1-radius, rowNumber+radius)]
neighbors(1, 2, 3)
## output
![image](https://user-images.githubusercontent.com/72538198/105152354-bab51a00-5b0f-11eb-9a9e-6657b8bfd883.png)

## for neighbours (3,2,3)
## output


