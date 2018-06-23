"""
File name: resize_training_images.py
Description: This file takes a path where exists the images for training 
and resizes to a new dimension (width, height) if the image was above 35x35
Command line example: python3 resize_training_images.py -p path_Images -d (width, height) 
Author: Renata Zottis Junges
Python Version: 3.6
"""

#python resize_training_images.py -p ../../60km_h/neg_pos_48x48/ 
#python resize_training_images.py -p ../../80km_h/neg_pos_48x48/ 

import numpy as np 
import cv2
import argparse
import glob

ap = argparse.ArgumentParser()
ap.add_argument('-p', "--path", required=True, help="path to the images that must be resized")
ap.add_argument('-d', "--dimension", required=False, default=(48,48), help="Dimension of training images (Width, Height)")
args = vars(ap.parse_args())

#Take the dimenson
width, height = args["dimension"] 

#Read all images in image directory
print("Resizing images...")
for file in glob.glob(args["path"] + "*.ppm"): #Put here the image extension
    file = file.replace('\\', '') #This treat if has some space in image name
    filename = file.split("/")[-1]
    
    img = cv2.imread(file)
    height_img, width_img, channel = img.shape
    
    if height_img >= 35 and width_img >= 35:
        img_resize = cv2.resize(img, (width, height))
        #Save the resized images in the  "pos" folder in the path directory (directory informed)
        cv2.imwrite(args["path"] + "pos/" + filename, img_resize)     
    