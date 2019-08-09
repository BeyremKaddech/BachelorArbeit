import os, sys
from time import sleep

i = 0

for filename in os.listdir("/Users/beyrem/Documents/Studies/BachelorArbeit/Code/faces+64/"):
    dst = str(i) + ".jpg"
    src = '/Users/beyrem/Documents/Studies/BachelorArbeit/Code/faces+64/' + filename
    dst = '/Users/beyrem/Documents/Studies/BachelorArbeit/Code/faces+64/' + dst
    # rename() function will
    # rename all the files
    os.rename(src, dst)
    i += 1
    print(filename)
    sleep(0.001)
print(i)
