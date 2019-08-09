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
print(imgs)

width = 128
height = 130
print('Resizing all images be %d pixels wide' % width)

folder = 'four64'
if not os.path.exists(folder):
    os.makedirs(folder)

# Iterate through resizing and saving
for img in imgs:
    pic = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    pic = cv2.resize(pic, (width, height))
    for i in range(2):
        for j in range(2):
            crop_pic = pic[1 + i*64:1 + (i+1)*64, j*64:(j+1)*64].copy()
            cv2.imwrite(folder + '/' + str(j) + 'x' + str(i) + img, crop_pic)