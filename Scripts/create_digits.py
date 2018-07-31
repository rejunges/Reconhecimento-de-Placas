"""
File name: create_digits.py
Description: This file creates digits samples to use in supervised learning. 
Author: Renata Zottis Junges
Python Version: 3.6
"""

import numpy as np 
import cv2
import os
import matplotlib.font_manager
import glob
import pickle
import imutils
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from imutils import paths, contours
import Augmentor
import shutil


directory = "../../datasets/digits/"

if not os.path.exists(directory):
	os.makedirs(directory)

for i in range(0,10):
	if not os.path.exists(directory + str(i) + "/"):
		os.makedirs(directory + str(i) + "/")

#Fonts in computer
#fonts = matplotlib.font_manager.findSystemFonts(fontpaths="/usr/share/fonts/truetype/", fontext='ttf')

#loads fonts installed in my computer 
file = open("fonts", 'rb')
fonts = pickle.load(file)
file.close()

#I don't want these fonts from fonts
list_font =  ['/usr/share/fonts/truetype/fonts-beng-extra/ani.ttf', '/usr/share/fonts/truetype/fonts-deva-extra/chandas1-2.ttf', '/usr/share/fonts/truetype/samyak/Samyak-Devanagari.ttf', '/usr/share/fonts/truetype/samyak-fonts/Samyak-Gujarati.ttf', '/usr/share/fonts/truetype/kacst/KacstNaskh.ttf', '/usr/share/fonts/truetype/kacst/KacstTitle.ttf', '/usr/share/fonts/truetype/fonts-gujr-extra/padmaa-Medium-0.5.ttf', '/usr/share/fonts/truetype/kacst/KacstPoster.ttf', '/usr/share/fonts/truetype/lohit-oriya/Lohit-Odia.ttf', '/usr/share/fonts/truetype/kacst/KacstDecorative.ttf', '/usr/share/fonts/truetype/samyak-fonts/Samyak-Tamil.ttf', '/usr/share/fonts/truetype/fonts-orya-extra/utkal.ttf', '/usr/share/fonts/truetype/kacst/KacstScreen.ttf', '/usr/share/fonts/truetype/kacst/KacstLetter.ttf', '/usr/share/fonts/truetype/kacst/KacstTitleL.ttf', '/usr/share/fonts/truetype/kacst/KacstDigital.ttf', '/usr/share/fonts/truetype/kacst/KacstArt.ttf', '/usr/share/fonts/truetype/kacst/KacstOffice.ttf', '/usr/share/fonts/truetype/kacst/KacstBook.ttf', '/usr/share/fonts/truetype/kacst/KacstPen.ttf', '/usr/share/fonts/truetype/kacst/KacstQurn.ttf', '/usr/share/fonts/truetype/kacst/KacstFarsi.ttf', '/usr/share/fonts/truetype/samyak-fonts/Samyak-Malayalam.ttf', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', '/usr/share/fonts/truetype/kacst/mry_KacstQurn.ttf', '/usr/share/fonts/truetype/sinhala/lklug.ttf']

cont = 0

for f in fonts:
	if f not in list_font:
		try:
			font = ImageFont.truetype(f,650)
		except:
			continue
		for number in range(0, 10):
			img=Image.new("RGB", (1000,1000),(0,0,0))
			draw = ImageDraw.Draw(img)
			draw.text((100, 0),str(number),(255,255,255),font=font)
			draw = ImageDraw.Draw(img)
			a = f.split("/")[-1].split(".")[0]
			
			img.save(directory + str(number) + "/" + str(cont) + "_" + str(a) + ".jpg")

			cont = cont + 1


#Opencv has only 8 fonts
for font in range(0, 8):
	
	for number in range (0, 10):
		for j in range (40, 110, 5):	
			img = np.zeros([1000,1000, 3], dtype = np.uint8) #create a black image
			if font == 1 or font == 5:
				img = cv2.putText(img, str(number), (150,800), font, 30*1.5, (255, 255, 255) , j)	
			else:
				img = cv2.putText(img, str(number), (150,800), font, 30, (255, 255, 255) , j)
			
			cv2.imwrite(directory + str(number) + "/" + str(cont) + ".jpg", img)
			cont = cont + 1

#Apply data augmentation with augmentor 
for i in range(0,10):
	p = Augmentor.Pipeline(directory + str(i))
	p.skew_left_right(probability=0.5, magnitude=0.1)
	p.random_distortion(probability=0.4, grid_width=2, grid_height=2, magnitude=2)
	p.rotate(probability=0.6, max_left_rotation=5, max_right_rotation=5)
	p.sample(1500)
	p.process()

#Put the data augmentation images in correct folder
for i in range(0, 10):
	current_directory = directory + str(i) + "/"
	files = glob.glob(current_directory +  "*.jpg")
	for f in files:
		os.remove(f)

	files = glob.glob(current_directory + "output/*") 
	for f in files:
		name = f.split("/")[-1]
		os.rename(f, current_directory + name)	

	shutil.rmtree(current_directory + "output")

#Read folders and resize images
for i in range(0,10):
	images = glob.glob(directory + str(i) + "/*")
	for path in images:
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		#find number contours 
		cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]

		#find the biggest bouding box
		mx = (0,0,0,0)      # biggest bounding box so far
		mx_area = 0
		for cont in cnts:
			x,y,w,h = cv2.boundingRect(cont)
			area = w*h
			if area > mx_area:
				mx = x,y,w,h
				mx_area = area
		x,y,w,h = mx

		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		new_img = img[y: y+h, x: x+w].copy()
		
		#resize
		new_img = cv2.resize(new_img, (48,48))
		cv2.imwrite(path, new_img)


