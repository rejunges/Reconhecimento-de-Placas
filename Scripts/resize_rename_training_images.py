"""
File name: resize_rename_training_images.py
Description: This file takes a path where exists folders with the images for training 
and resizes to a new dimension (width, height) if the image was above 35x35. Names should be 
autoincrement int.
Command line example: python3 resize_rename_training_images.py -p path_Images -d (width, height) 
Author: Renata Zottis Junges
Python Version: 3.6
"""

#python resize_rename_training_images.py -p ../../datasets/trafficSign/pos/

import numpy as np 
import os
import cv2
import argparse
import glob
import shutil

ap = argparse.ArgumentParser()
ap.add_argument('-p', "--path", required=True, help="path to the images that must be resized")
ap.add_argument('-d', "--dimension", required=False, default=(48,48), help="Dimension of training images (Width, Height)")
args = vars(ap.parse_args())

#Take the dimenson
width, height = args["dimension"] 

path = args["path"]
#Take all folders from this path
list_folders = next(os.walk(path))[1]

filename = 0
for folder in list_folders:
	#Read all images in image directory
	print("Resizing images from " + folder)
	os.makedirs(path + folder + "-copia")
	for file in glob.glob( path + folder  + "/*.ppm"): #Put here the image extension
		file = file.replace('\\', '') #This treat if has some space in image name
		filename += 1
		img = cv2.imread(file)
		height_img, width_img, channel = img.shape
		
		if height_img >= 35 and width_img >= 35:
			img_resize = cv2.resize(img, (width, height))
			#Save the resized images in the path directory (directory informed)
			cv2.imwrite(path + folder + "-copia/" + str(filename) + ".jpg", img_resize)     
	shutil.rmtree(path + folder) #remove folder