#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:34:48 2020

@author: petar
"""

import cv2 as cv
import numpy as np

def get_cube_image(i, maskOut, homeDir):
    img = cv.imread(homeDir)
    
    black_level = 2048
    saturation_level = np.max(img[:]) - 2
    
    img = img - black_level
    img[img < 0] = 0
    
    m = np.zeros([img.shape[0], img.shape[1]])
    for ch in range(3):
        m = m + (img[:, :, ch] >= saturation_level - black_level).astype(float)
        print(m)

    m = m > 0
    
    if maskOut != 0:
        m[1050:, 2050:] = 1
        
    for ch in range(3):
        channel = img[:, :, ch]
        print(m, "m")
        print(channel)
        channel[m] = 0
        img[:, :, ch] = channel
        
    return img
