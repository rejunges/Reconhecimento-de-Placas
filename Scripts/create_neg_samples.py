"""
File name: create_neg_samples.py
Description: This file creates negative samples to use in supervised learning.
It is necessary to inform a path to the video to cut the images from frames
Command line example: python3 create_neg_samples.py -v video 
Author: Renata Zottis Junges
Python Version: 3.6
"""

#python create_neg_samples.py -v ../../vts/clip_i5s_0094.MOV
 
import numpy as np 
import cv2
import argparse
import glob
import sys

ap = argparse.ArgumentParser()
ap.add_argument('-v', "--video", required=True, help="path to the video")
ap.add_argument('-d', "--dimension", required=False, default=(48,48), help="Dimension of training images (Width, Height)")
args = vars(ap.parse_args())

#Take the dimenson
width, height = args["dimension"] 

#Read all images in image directory
print("Creating negative samples")
cap = cv2.VideoCapture(args["video"]) #Capture the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
j = 0
cont = 0

for frame in range(0, total_frames):
    #for each frame cut the image in pieces of 48x48    
    ret, frame = cap.read() #Capture frame-by-frame    
    width_frame, height_frame = frame.shape[1], frame.shape[0]

    w = 0
    h = 500 #start in 500 px to reduce the sky area
    cont = cont + 1 
    
    #jump frames to reduce redundancy 
    if cont % 170 == 1:
        while h + height < height_frame:
            if (w + width > width_frame):
                h = h + height
                w = 0
            rect = frame[h: h + height, w: w + width].copy()
            w = w + width
            
            if rect.shape[1] == width and rect.shape[0] == height:
                cv2.imwrite('Neg/' + str(j) +'.jpg', rect) #The Neg folder should be exist in current directory
                j = j + 1
            
            if j == 2000: #Here put the number of negative samples you want 
                sys.exit(0)