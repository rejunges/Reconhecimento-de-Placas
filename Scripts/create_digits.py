"""
File name: create_digits.py
Description: This file creates digits samples to use in supervised learning. 
Author: Renata Zottis Junges
Python Version: 3.6
"""

import numpy as np 
import cv2
import os

directory = "../../datasets/digits/"
if not os.path.exists(directory):
	os.makedirs(directory)

for i in range(0,10):
	if not os.path.exists(directory + str(i) + "/"):
		os.makedirs(directory + str(i) + "/")


#Opencv has only 8 fonts
cont = 0
for font in range(0, 8):
	
	for number in range (0, 10):
		for j in range (40, 110, 5):	
			img = np.zeros([1000,1000, 3], dtype = np.uint8) #create a black image
			if font == 1 or font == 5:
				img = cv2.putText(img, str(number), (150,800), font, 30*1.5, (255, 255, 255) , j)	
			else:
				img = cv2.putText(img, str(number), (150,800), font, 30, (255, 255, 255) , j)

			
			img = cv2.resize(img, (28,28))
			cv2.imwrite(directory + str(number) + "/" + str(cont) + ".jpg", img)
			cont = cont + 1