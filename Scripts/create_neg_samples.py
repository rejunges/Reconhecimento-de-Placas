"""
File name: create_neg_samples.py
Description: This file creates negative samples to use in supervised learning.
It is necessary to inform a path to the video to cut the images from frames
Command line example: python3 create_neg_samples.py -v video 
Author: Renata Zottis Junges
Python Version: 3.6
"""

#python create_neg_samples.py -v ../../vts/80km/img_5169.mov
#python create_neg_samples.py -v ../../sinalizacao_vertical/4.mov
 
import numpy as np 
import cv2
import argparse
import glob
import sys
import random

ap = argparse.ArgumentParser()
ap.add_argument('-v', "--video", required=True, help="path to the video")
ap.add_argument('-d', "--dimension", required=False, default=(48,48), help="Dimension of training images (Width, Height)")
args = vars(ap.parse_args())
seed = 9

#Take the dimenson
width, height = args["dimension"] 

#Read all images in image directory
print("Creating negative samples")
cap = cv2.VideoCapture(args["video"]) #Capture the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
j = 0
cont = 0
print((total_frames/60) * 10)

random.seed(seed)
for frame in range(0, total_frames):
    #for each frame cut the image in pieces of 48x48    
    ret, frame = cap.read() #Capture frame-by-frame    
    width_frame, height_frame = frame.shape[1], frame.shape[0]

    w_initial, w_final = 0, width_frame - 200 
    # start in 500 px to reduce the sky area
    h_inicial, h_final = 500, height_frame - 200

    cont = cont + 1 
    
    #jump frames to reduce redundancy (2 sec)
    if cont % 60 == 1:
        for i in range(0,10):
            w = random.randrange(w_initial, w_final)
            h = random.randrange(h_inicial, h_final)
            
            rect = frame[h: h + 200, w: w + 200].copy()
            rect = cv2.resize(rect, (48,48))
            cv2.imwrite("neg/" + str(j) + ".jpg", rect)
            j += 1
            