# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:50:36 2019

Code works for asked all filters and images.
The outputs in the report are generated using this code
However, few examples with different filters are shown in the code
to reduce the code size for processing
uncommenting the python code can work in various desired ways

It only uses lena img, to use wolves img, change the imread function value.

@author: Vivek Rathi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# convert float to uint8
def ftou8(img):
    img = img.astype(np.uint8)
    return img

# convert uint8 to float
def u8tof(img):
    img = img.astype(np.float32)
    return img

# spread intensities
def spread_linear(img,uplimit):
    s = np.copy(img)
    ma = img.max()
    mi = img.min()
    if len(img.shape) == 3:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                s[i,j,:] = ((uplimit - 1)/(ma - mi)) * (img[i,j,:] - mi)
        return (ftou8(s))
    else:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                s[i,j] = ((uplimit - 1)/(ma - mi)) * (img[i,j] - mi)
        return (ftou8(s))
#padding function
def pad(img,typepad):
    rows = img.shape[0]
    cols = img.shape[1]
    if len(img.shape) == 3:
        imgpad = np.zeros((rows+2,cols+2,3), dtype = 'float')
    elif len(img.shape) == 2:
        imgpad = np.zeros((rows+2,cols+2), dtype = 'float')
    imgpad[1:rows+1,1:cols+1] = img
    rows = imgpad.shape[0]
    cols = imgpad.shape[1]
    if typepad == "zero":
        return imgpad
    elif typepad == "wrap":
        imgpad[:,0] = imgpad[:,cols-2]
        imgpad[:,cols-1] = imgpad[:,1]
        imgpad[0,:] = imgpad[rows-2,:]
        imgpad[rows-1,:] = imgpad[1,:]
        imgpad[0,0] = imgpad[rows-2,cols-2]
        imgpad[rows-1,0] = imgpad[1,cols-2]
        imgpad[0,cols-1] = imgpad[rows-2,1]
        imgpad[rows-1,cols-1] = imgpad[1,1]
        return imgpad
    elif typepad == "copyedge":
        imgpad[0,:] = imgpad[1,:]
        imgpad[rows-1,:] = imgpad[rows-2,:]
        imgpad[:,0] = imgpad[:,1]
        imgpad[:,cols-1] = imgpad[:,cols-2]
        imgpad[0,0] = imgpad[1,1]
        imgpad[rows-1,0] = imgpad[rows-2,1]
        imgpad[0,cols-1] = imgpad[1,cols-2]
        imgpad[rows-1,cols-1] = imgpad[rows-2,cols-2]
        return imgpad
    elif typepad == "reflect":
        imgpad[0,:] = imgpad[2,:]
        imgpad[rows-1,:] = imgpad[rows-3,:]
        imgpad[:,0] = imgpad[:,2]
        imgpad[:,cols-1] = imgpad[:,cols-3]
        imgpad[0,0] = imgpad[2,2]
        imgpad[rows-1,0] = imgpad[rows-3,2]
        imgpad[0,cols-1] = imgpad[2,cols-3]
        imgpad[rows-1,cols-1] = imgpad[rows-3,cols-3]
        return imgpad
    
# convolution

'''
p = 0 -> zero padding
p = 1 -> wrap padding
p = 2 -> copyedge padding
p = 3 -> reflect padding
'''
def conv(img,k,p):
    imgcopy = img.astype(np.float32)
    if p == 0:
        padimg = pad(img,"zero")
    elif p == 1:
        padimg = pad(img,"wrap")
    elif p == 2:
        padimg = pad(img,"copyedge")
    elif p == 3:
        padimg = pad(img,"reflect")
        
    if len(img.shape) == 3:
        B = np.zeros((k.shape[0],k.shape[1],3))
        r = padimg.shape[0]
        c = padimg.shape[1]
        if k.shape[0] == k.shape[1] == 3:
            for i in range(r-2):
                for j in range(c-2):
                    B[:,:,0] = np.multiply(k,padimg[i:i+3,j:j+3,0])
                    B[:,:,1] = np.multiply(k,padimg[i:i+3,j:j+3,1])
                    B[:,:,2] = np.multiply(k,padimg[i:i+3,j:j+3,2])
                    imgcopy[i,j,0] = np.sum(B[:,:,0])
                    imgcopy[i,j,1] = np.sum(B[:,:,1])
                    imgcopy[i,j,2] = np.sum(B[:,:,2])
            return imgcopy
        elif k.shape[0] == k.shape[1] == 2:
            if k.all == k_rx.all():
                for i in range(r-2):
                    for j in range(c-2):
                        B[:,:,0] = np.multiply(k,padimg[i+1:i+3,j:j+2,0])
                        B[:,:,1] = np.multiply(k,padimg[i+1:i+3,j:j+2,1])
                        B[:,:,2] = np.multiply(k,padimg[i+1:i+3,j:j+2,2])
                        imgcopy[i,j,0] = np.sum(B[:,:,0])
                        imgcopy[i,j,1] = np.sum(B[:,:,1])
                        imgcopy[i,j,2] = np.sum(B[:,:,2])
                return imgcopy
            else:
                for i in range(r-2):
                    for j in range(c-2):
                        B[:,:,0] = np.multiply(k,padimg[i+1:i+3,j+1:j+3,0])
                        B[:,:,1] = np.multiply(k,padimg[i+1:i+3,j+1:j+3,1])
                        B[:,:,2] = np.multiply(k,padimg[i+1:i+3,j+1:j+3,2])
                        imgcopy[i,j,0] = np.sum(B[:,:,0])
                        imgcopy[i,j,1] = np.sum(B[:,:,1])
                        imgcopy[i,j,2] = np.sum(B[:,:,2])
                return imgcopy
                
        elif k.shape[0] < k.shape[1]:
            for i in range(r-2):
                for j in range(c-2):
                    B[:,:,0] = np.multiply(k,padimg[i+1:i+2,j+1:j+3,0])
                    B[:,:,1] = np.multiply(k,padimg[i+1:i+2,j+1:j+3,1])
                    B[:,:,2] = np.multiply(k,padimg[i+1:i+2,j+1:j+3,2])
                    imgcopy[i,j,0] = np.sum(B[:,:,0])
                    imgcopy[i,j,1] = np.sum(B[:,:,1])
                    imgcopy[i,j,2] = np.sum(B[:,:,2])
            return imgcopy
        elif k.shape[0] > k.shape[1]:
            for i in range(r-2):
                for j in range(c-2):
                    B[:,:,0] = np.multiply(k,padimg[i+1:i+3,j+1:j+2,0])
                    B[:,:,1] = np.multiply(k,padimg[i+1:i+3,j+1:j+2,1])
                    B[:,:,2] = np.multiply(k,padimg[i+1:i+3,j+1:j+2,2])
                    imgcopy[i,j,0] = np.sum(B[:,:,0])
                    imgcopy[i,j,1] = np.sum(B[:,:,1])
                    imgcopy[i,j,2] = np.sum(B[:,:,2])
            return imgcopy
    elif len(img.shape) == 2:
        B = np.zeros((k.shape[0],k.shape[1]))
        r = padimg.shape[0]
        c = padimg.shape[1]
        if k.shape[0] == k.shape[1] == 3:
            for i in range(r-2):
                for j in range(c-2):
                    B = np.multiply(k,padimg[i:i+3,j:j+3])
                    imgcopy[i,j] = np.sum(B)
            return imgcopy
        elif k.shape[0] == k.shape[1] == 2:
            if k.all() == k_rx.all():
                for i in range(r-2):
                    for j in range(c-2):
                        B = np.multiply(k,padimg[i+1:i+3,j:j+2])
                        imgcopy[i,j] = np.sum(B)
                return imgcopy
            else:
                for i in range(r-2):
                    for j in range(c-2):
                        B = np.multiply(k,padimg[i+1:i+3,j+1:j+3])
                        imgcopy[i,j] = np.sum(B)
                return imgcopy
                
        elif k.shape[0] < k.shape[1]:
            for i in range(r-2):
                for j in range(c-2):
                    B = np.multiply(k,padimg[i+1:i+2,j+1:j+3])
                    imgcopy[i,j] = np.sum(B)
            return imgcopy
        elif k.shape[0] > k.shape[1]:
            for i in range(r-2):
                for j in range(c-2):
                    B = np.multiply(k,padimg[i+1:i+3,j+1:j+2])
                    imgcopy[i,j] = np.sum(B)
            return imgcopy

'''
Kernels
'''
k_box = (1/9) * np.ones((3,3))
k_x1 = np.matrix(([-1,1]))
k_y1 = np.matrix(([-1],[1]))
k_px = np.matrix(([-1,0,1],[-1,0,1],[-1,0,1]))
k_py = np.matrix(([1,1,1],[0,0,0],[-1,-1,-1]))
k_sx = np.matrix(([-1,0,1],[-2,0,2],[-1,0,1]))
k_sy = np.matrix(([1,2,1],[0,0,0],[-1,-2,-1]))
k_rx = np.matrix(([0,1],[-1,0]))
k_ry = np.matrix(([1,0],[0,-1]))

# Load image
lenaimg = cv2.imread('lena.png',0)
#lenaimg = cv2.imread('lena.png',1)

# Apply Convolution
# Box Filter- Zero Padding
c11 = conv(lenaimg,k_box,0)
s11 = ftou8(c11)
# Fisrt Order X Derivative- Wrap Padding
d11 = conv(lenaimg,k_x1,1)
s12 = spread_linear(d11,256)
# Prewitt X Derivative - Copyedge Padding
e11 = conv(lenaimg,k_px,2)
s13 = spread_linear(e11,256)
# Sobel Y Derivative - Reflect Padding
f11 = conv(lenaimg,k_sy,3)
s14 = spread_linear(f11,256)
# Roberts Y Derivative - Copyedge Padding
g11 = conv(lenaimg,k_ry,2)
s15 = spread_linear(g11,256)



plt.figure(figsize=(10,10))

plt.subplot(221)
#plt.imshow(cv2.cvtColor(s11, cv2.COLOR_BGR2RGB))
plt.imshow(s11, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Box Filter - Pad 0')

plt.subplot(222)
#plt.imshow(cv2.cvtColor(s12, cv2.COLOR_BGR2RGB))
plt.imshow(s12, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('X Derivative - Wrap')

plt.subplot(223)
plt.imshow(s13, cmap = 'gray')
#plt.imshow(cv2.cvtColor(s13, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.title('Prewitt X - Copy')

plt.subplot(224)
#plt.imshow(cv2.cvtColor(s14, cv2.COLOR_BGR2RGB))
plt.imshow(s14, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Sobel Y - Reflect')
plt.show()



cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image',lenaimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.namedWindow('Rx Derivative-Pad 0', cv2.WINDOW_NORMAL)
#cv2.imshow('Rx Derivative-Pad 0',c11)
#cv2.namedWindow('Rx Derivative-Pad wrap', cv2.WINDOW_NORMAL)
#cv2.imshow('Rx Derivative-Pad wrap',d11)
#cv2.namedWindow('Rx Derivative-Pad copy', cv2.WINDOW_NORMAL)
#cv2.imshow('Rx Derivative-Pad copy',e11)
#cv2.namedWindow('Rx Derivative-Pad reflect', cv2.WINDOW_NORMAL)
#cv2.imshow('Rx Derivative-Pad reflect',f11)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


