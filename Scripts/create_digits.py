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
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

directory = "../../datasets/digits/"
if not os.path.exists(directory):
	os.makedirs(directory)

for i in range(0,10):
	if not os.path.exists(directory + str(i) + "/"):
		os.makedirs(directory + str(i) + "/")

fonts = matplotlib.font_manager.findSystemFonts(fontpaths="/usr/share/fonts/truetype/", fontext='ttf')
wrong_fonts = ["KacstNaskh.tff", "KacstOffice.tff", 'Samyak-Malayalam.ttf', 'lklug.ttf' , "Samyak-Gujarati.ttf", "KacstFarsi.ttf", "KacstQurn.ttf", "KacstQurn.ttf",  "KacstPen.ttf", "DroidSansFallbackFull.ttf", "KacstFarsi.ttf" , "Samyak-Tamil.ttf", "KacstTitleL.ttf", "mry_KacstQurn.ttf", "KacstDecorative.ttf", "Samyak-Malayalam.ttf", "Samyak-Tamil.ttf", "KacstNaskh.ttf", "KacstTitle.ttf", "Lohit-Odia.ttf", "KacstLetter.ttf", "DroidSansFallbackFull.ttf", "KacstDecorative.ttf", "KacstDigital.ttf" , "KacstOffice.ttf", "KacstPen.ttf", "KacstArt.ttf", "utkal.ttf", "Lohit-Odia.ttf", "KacstDigital.ttf", "KacstBook.ttf", "Samyak-Devanagari.ttf", "KacstScreen.ttf", "KacstPoster.ttf", "KacstArt.ttf", "padmaa-Medium-0.5.ttf", "KacstBook.ttf", "KacstLetter.ttf", "padmaa-Medium-0.5.ttf", "KacstTitleL.ttf", "utkal.ttf", "lklug.ttf", "KacstTitle.ttf", "mry_KacstQurn.ttf", "Samyak-Devanagari.ttf", "Samyak-Gujarati.ttf", 'KacstPoster.ttf', "KacstScreen.ttf" ]
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

#Read folders and resize images
for i in range(0,10):
	images = glob.glob(directory + str(i) + "/*")
	for path in images:
		img = cv2.imread(path)
		#resize
		img = cv2.resize(img, (28,28))
		cv2.imwrite(path, img)