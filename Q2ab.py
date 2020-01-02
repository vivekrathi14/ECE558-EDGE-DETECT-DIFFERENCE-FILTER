# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:05:29 2019

@author: Vivek Rathi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

'''
DFT2 - calculate 2d fft of i/p
'''
def DFT2(img):
    if (img.dtype != complex):
        fimg = img.astype(np.float)
    else:
        fimg = np.copy(img)
    fimg = fimg / fimg.max()
    fftx = np.fft.fft(fimg,axis=0)
    fftxy = np.fft.fft(fftx,axis=1)
    return fftxy

'''
getimg - gives o/p as uint8 
'''
def getimg(g,img):
    v = abs(g) * img.max()
    v = np.round(v).astype(np.uint8)
    return v
'''
shift - shifts fourier spectrum to the centre, i.e. low frequencies
'''
def shift(img): 
    h = img.astype(np.float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h[i,j] = img[i,j]* (-1)**((i+j))
    return h
'''
IDFT2 - inv fft using fft
'''
def IDFT2(fft2d):
    f_conj = np.conj(fft2d)
    ffxy = DFT2(f_conj)
    ffxy = ffxy / (fft2d.shape[0]*fft2d.shape[1])
    ffxy = ffxy / (abs(ffxy)).max()
    #im = np.conj(ffxy)
    im = np.conj(ffxy)
    return im
# read image
limg = cv2.imread('wolves.png',0)
lshift = shift(limg)
fl = DFT2(lshift)

#wimg = cv2.imread('wolves.png',0)
#wshift = shift(wimg)
#fw = DFT2(lshift)

# magnitude and phase spectrum
ms_l = np.log(1+np.abs(fl))

ps_l = np.angle(fl)

# IDFT
gl = IDFT2(fl)
img_gl = getimg(gl,limg)

# difference
d_l = limg - img_gl

# Plots

plt.figure(figsize=(10,10))

plt.subplot(221)
plt.imshow(limg, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Input Image')

plt.subplot(222)
plt.imshow(ms_l, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Magnitude Spectrum')

plt.subplot(223)
plt.imshow(ps_l, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Phase Spectrum')

plt.subplot(224)
plt.imshow(d_l, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Difference')
plt.show()




        