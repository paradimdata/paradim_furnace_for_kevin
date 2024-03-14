import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

from google.cloud import vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'Image Treatment-5b3bb079ba4e.json'
from SciServer import CasJobs, Files, Authentication

casjobs_context='MyDB'
subject_name="Ali Rachidi"
    
class landmarkedimage:
    def __init__(self, frameName, frame):
        self.frameName = frameName
        self.frame = frame

    def print_landmarkedimage(self):
        print("Frame Number: " + self.frameName + '\n')
        print("presentValue: " + str(self.presentValue) + '\n')
        print("setValue: " + str(self.setValue) + '\n')
        print("upperShaftRotation: " + str(self.upperShaftRotation) + '\n')
        print("lowerShaftRotation: " + str(self.lowerShaftRotation) + '\n')
        print("upperShaftFast: " + str(self.upperShaftFast) + '\n')
        print("upperShaftSlow: " + str(self.upperShaftSlow) + '\n')
        print("lowerShaftFast: " + str(self.lowerShaftFast) + '\n')
        print("lowerShaftSlow: " + str(self.lowerShaftSlow) + '\n')
        print("temperature: " + str(self.temperature) + '\n')

    frameName = ''
    frame = ''
    points = dict(list())
    presentValue = 0.0
    setValue = 0.0
    upperShaftRotation = 0.0
    lowerShaftRotation = 0.0
    upperShaftFast = 0.0
    upperShaftSlow = 0.0
    lowerShaftFast = 0.0
    lowerShaftSlow = 0.0
    temperature = 0.0
    
def detect_text(path):
   
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    print(type(content))
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description

import matplotlib.pyplot as plt

def notebook_display_img(img):
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(img)
    plt.show()
    
def frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder):
    
    img = cv2.imread(str(frameFolder + '/' + file))
    width, height, pixels = img.shape
    crop_img = img[(int)(width*yStartPct):(int)(width*yStopPct),(int)(height*xStartPct):(int)(height*xStopPct)].copy()
    notebook_display_img(crop_img)
    return crop_img

print("start")

frameFolder = 'frameFolder' # Folder destination where the frames should be saved

files = [f for f in listdir(frameFolder) if isfile(join(frameFolder, f))]

i = 0

for file in files:
    
    frameName = "frame" + str(i)
    path = 'frames_destinations/' + frameName + '.jpg'
    
    # frame 
    xStartPct = .65 
    xStopPct = 0.95
    yStartPct = .1
    yStopPct = .5
    
    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    cv2.imwrite(path, frame)
    li = landmarkedimage(frameName, frame) # initiate object

    # Present Value
    xStartPct = .69
    xStopPct = 0.73
    yStartPct = .53
    yStopPct = .56

    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    presentValue = detect_text(path)
    li.presentValue = float(presentValue)
    
    # Set Value
    xStartPct = .69
    xStopPct = 0.73
    yStartPct = .56
    yStopPct = .6

    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    setValue = detect_text(path)
    li.setValue = float(setValue)
    
    # Upper Shaft Slow Moving 
    xStartPct = .50 # Paramters of the cropped image in percentage of the frame
    xStopPct = 0.53
    yStartPct = .15
    yStopPct = .18

    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    upperShaftSlow = detect_text(path)
    li.upperShaftSlow = float(upperShaftSlow)
    
    # Upper Shaft Fast Moving 
    xStartPct = .60
    xStopPct = 0.63
    yStartPct = .15
    yStopPct = .18

    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    upperShaftFast = detect_text(path)
    li.upperShaftFast = float(upperShaftFast)
    
    # Lower Shaft Slow Moving 
    xStartPct = .50 # Paramters of the cropped image in percentage of the frame
    xStopPct = 0.53
    yStartPct = .41
    yStopPct = .44
    
    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    lowerShaftSlow = detect_text(path)
    li.lowerShaftSlow = float(lowerShaftSlow)
    
    # Lower Shaft Fast Moving 
    xStartPct = .60
    xStopPct = 0.63
    yStartPct = .41
    yStopPct = .44

    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    lowerShaftFast = detect_text(path)
    li.lowerShaftFast = float(lowerShaftFast)
    
    # upper Shaft Rotation
    xStartPct = .4
    xStopPct = 0.43
    yStartPct = .15
    yStopPct = .18

    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    upperShaftRotation = detect_text(path)
    li.upperShaftRotation = float(upperShaftRotation)
    
    # lower Shaft Rotation
    xStartPct = .4
    xStopPct = 0.43
    yStartPct = .41
    yStopPct = .44

    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    lowerShaftRotation = detect_text(path)
    li.lowerShaftRotation = float(lowerShaftRotation)
    
    # Temperature
    xStartPct = .225 # Paramters of the cropped image in percentage of the frame
    xStopPct = 0.255
    yStartPct = .67
    yStopPct = .695
    
    frame = frameExtractor(xStartPct, xStopPct, yStartPct, yStopPct, file, frameFolder)
    path = 'temp.jpg'
    cv2.imwrite(path, frame)
    temperature = detect_text(path)
    li.temperature = float(temperature)
    
    li.print_landmarkedimage()
    
    insert_query = '''INSERT INTO furnaceStateHistory
                (presentValue, setValue, upperShaftRotation, lowerShaftRotation, upperShaftFast, upperShaftSlow, 
                lowerShaftFast, lowerShaftSlow)
                VALUES
                ('{0}', '{1}','{2}', '{3}', '{4}', '{5}', '{6}', '{7}')'''.format(li.presentValue, li.setValue,
                li.upperShaftRotation, li.lowerShaftRotation, li.upperShaftFast, li.upperShaftSlow, li.lowerShaftFast, 
                li.lowerShaftSlow)
    CasJobs.executeQuery(sql=insert_query, context=casjobs_context)
    
    i += 1
