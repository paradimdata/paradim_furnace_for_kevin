import matplotlib.pyplot as plt
import cv2
import os
from os import listdir
from os.path import isfile, join


def notebook_display_img(img):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(img)
    plt.show()
    
xStartPct = .65 
xStopPct = 0.95
yStartPct = .1
yStopPct = .5

image_destination = 'full_images' # Folder destination where the frames should be saved

files = [f for f in listdir('frameFolder') if isfile(join('frameFolder', f))]

for file in files:
    img = cv2.imread('frameFolder/' + file)
    width, height, pixels = img.shape
    crop_img = img[(int)(width*yStartPct):(int)(width*yStopPct),(int)(height*xStartPct):(int)(height*xStopPct)].copy()
    notebook_display_img(crop_img)
    break
