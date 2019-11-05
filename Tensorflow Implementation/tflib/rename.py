import os, sys

i = 0

for filename in os.listdir("/Users/beyrem/Documents/Studies/BachelorArbeit/Code/faces64/"):
    dst = str(i) + ".jpg"
    src = '/Users/beyrem/Documents/Studies/BachelorArbeit/Code/faces64/' + filename
    dst = '/Users/beyrem/Documents/Studies/BachelorArbeit/Code/faces64/' + dst
    # rename() function will
    # rename all the files
    os.rename(src, dst)
    i += 1