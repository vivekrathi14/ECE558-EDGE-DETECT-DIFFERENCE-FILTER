# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:40:37 2019

@author: Vivek Rathi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# convolution function
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

#create image
img = np.zeros((1024,1024))
img[511,511] = 1

imgc = conv(img,k_box,0)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image',img)
cv2.namedWindow('Convolved Image', cv2.WINDOW_NORMAL)
cv2.imshow('Convolved Image',imgc)
cv2.waitKey(0)
cv2.destroyAllWindows()