#!/usr/bin/env python
__author__ = 'Alex Galea'

''' Script to resize all files in current directory,
    saving new .jpg and .jpeg images in a new folder. '''

import cv2
import glob
import os

# Get images
imgs = glob.glob('*.jpg')
imgs.extend(glob.glob('*.jpeg'))

print('Found files:')
height = 28
width = 28
print('Resizing all images be %d pixels wide' % width)

folder = 'resize'
if not os.path.exists(folder):
    os.makedirs(folder)

# Iterate through resizing and saving
for img in imgs:
    pic = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    pic = cv2.resize(pic, (width, height))
    cv2.imwrite(folder + '/' + img, pic)