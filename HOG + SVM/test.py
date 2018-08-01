"""
File name: test.py
Description: This file identifies and recognizes traffic signs in videos
Author: Renata Zottis Junges
Python Version: 3.6
"""

#python test.py -t ../../vts/60km/ 
#python test.py -t ../../sinalizacao_vertical/ 

import numpy as np 
import pickle
import cv2
import argparse
import glob
import time
from skimage.feature import hog
from sklearn import svm
from imutils import paths, contours
import imutils
from imutils import contours
import os
import helpers

def add_temp_coherence(detected_sign, recognized_sign, coord1=None, coord2=None):
	""" Put new list in tempo_coherence list 
	Args:
		detected_sign (bool): true if exists a sign in frame otherwise false
		recognized_sign (str): name of recognized sign otherwise None
	"""

	ant_temp = temp_coherence.pop() #remove the same frame informing 1 in detected sign
	fn, ds, rs, c1, c2 = ant_temp[-1] #take the last one
	if fn == frame_number:
#		print("temp_coherence: ",temp_coherence)
		if ds == False: #Before traffic sign identification
			atual = [frame_number, detected_sign, recognized_sign, coord1, coord2]
			temp_coherence.append([atual])
		else:
			if recognized_sign == None: #after the first traffic sign recognizion
				atual = [frame_number, detected_sign, recognized_sign, coord1, coord2]	
				ant_temp.append(atual)
				temp_coherence.append(ant_temp)
			elif rs == None:
				#iterate over the list to replace the recognized_sign
				l_final = []
				flag = True
				for l in ant_temp:
					fn, ds, rs, c1, c2 = l
					if rs == None and flag:
						atual = [frame_number, detected_sign, recognized_sign, c1, c2]
						l_final.append(atual)
						flag = False
					else:
						l_final.append(l)
				temp_coherence.append(l_final)
						 
			else:
				atual = [frame_number, detected_sign, recognized_sign, c1, c2]
				ant_temp.append(atual)
				temp_coherence.append(ant_temp)
				
#		print("\n")
#		print("temp_coherence2: ",temp_coherence)
#		print("\n")

def open_model(model, path = "../Models/"):
	"""Open model through picle

	Args:
		model (str): model from the training fase saved in pickle 
	
	Returns:
		model: training model

	"""
	
	file = open(path + model, 'rb')
	model = pickle.load(file)
	file.close()

	return model
	

def load_models():
	"""Load the traffic signs, no overtaking and digits models 
	
	Returns:
		tuple: returns a tuple with 3 elements, the traffic sign, no overtaking and digits models
	
	"""

	traffic_signs_model = open_model("trafficSigns.dat")  #first model is trafficSigns.dat
	no_overtaking_model = open_model("noOvertaking.dat")	#second model is noOvertaking.dat
	digits_model = open_model("digits5.dat")	#third model is digits.dat

	return traffic_signs_model, no_overtaking_model, digits_model

def preprocessing_frame(frame):
	"""Preprocessing the frame before goes to Hough Circle and predictions.
	This function the uses clahe filter and creates a binary mask with red color from the frame

	Args:
		frame (numpy.ndarray): frame from the video

	Returns:
		tuple: returns a tuple with 2 elements, the binary mask and the image after clahe  

	"""

	img = helpers.clahe(frame)	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)	#Put the image in HSV color space
	mask = helpers.mask_hsv_red(hsv)	#mask with the red color of the img
	mask = cv2.GaussianBlur(mask, (3,3), 0) 
	
	return mask, img

def preprocessing_speed_limit(rect):
	''' Creates a binary mask to segments the digits from speed limit signs.

	Args:
		rect (numpy.ndarray): speed limit image from original frame (without resize)

	Returns:
		(numpy.ndarray): binary image from preprocessing
	'''

	#Take rect because rect has not been resized
	img_gray = cv2.cvtColor(rect.copy(), cv2.COLOR_BGR2GRAY)
#	img_gray = cv2.GaussianBlur(img_gray, (5,5), 0) #REUNIAO:Rever (3,3) ou nada

	# Threshold the image
	ret, img_gray = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV) #REUNIAO aumenta 90-testando (10 em 10) até 128+-

	return img_gray

def predict_speed_limit_sign(final_frame, rect, model, dimensions):
	'''Segments rect image in digits and checks the value (0-9) for each digit using the
	digits model. 

	Args:
		final_frame (numpy.ndarray): image to draw the putText
		rect (numpy.ndarray): image rectangle (ROI) to verify which speed limit sign it is
		model: digits model from train.py
		dimensions (tuple): width and heigh to resize the digits image
	
	Returns:
		numpy.ndarray: frame with text
	
	'''

	img_gray = preprocessing_speed_limit(rect)
	
	# Find contours in binary image
	cnts = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #REUNIAO: trocar cadeia simples por outros
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	digitCnts = []
	
	img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
	teste = img_gray.copy() #DEBUG
	
	#loop over the candidates of digit area 
	for c in cnts:
		#compute the bounding box 
		(x, y, w, h) = cv2.boundingRect(c)
		
		img_gray_w, img_gray_h = img_gray.shape[1], img_gray.shape[0]
		if w >= img_gray_w/4 and x > 3 and (x + w < img_gray_w - img_gray_w/10) and y > 3 and y < img_gray_h:
			#DEBUG
			"""
			if frame_number > 749 and frame_number < 815: #placa de 80km
				cv2.rectangle(teste, (x ,y), (x+w,y+h), (0,0,255), 2)
				cv2.imwrite("result/" + str(frame_number) + "-3digits-bb-" + str(contador) + ".jpg", teste) 
			"""	
			digitCnts.append(c)

	#sort the contours from left-to-right
	if digitCnts:
		digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
	
	digits = ""
	img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
	
	#loop over each of digits:
	for c in digitCnts:
		(x, y, w, h) = cv2.boundingRect(c)
		roi = rect[y : y+h, x : x+w ] #extract the digit ROI

		roi = cv2.resize(roi, (48,48)) #resize to HOG 
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		ret, roi = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY_INV)
		#HOG method
		(H, hogImage) = hog(roi, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=True, block_norm='L2-Hys')
		
		#predict the image based on model 
		digits_pred = digits_model.predict(H.reshape(1,-1))[0]
		
		digit = (digits_pred.title()).lower()
		#DEBUG
		"""
		if frame_number > 749 and frame_number < 815: #placa de 80km
			cv2.imwrite("result/" + str(frame_number) + "-4digit-" + str(digit) + ".jpg", roi) 
		"""	
		#DEBUG
		"""
		if digit == "8":
			print("OITO ", frame_number)
		"""

		if digit == "1":
			digits = digits + "1"
			#continue reading because can be 10, 100, 120, 125
		elif len(digits) > 0: #If the first number is 1 then the need to read the others
			digits = digits + digit
		else: 
			add_temp_coherence(True, str(digit)+"0")
			cv2.putText(final_frame, "Recognized: " + digit + "0 km/h",(10,350), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
			break

	if len(digits) > 0:
		add_temp_coherence(True, str(digits))
		cv2.putText(final_frame, "Recognized: " + digits + " km/h",(10,350), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
	
	return final_frame

def predict_no_overtaking_sign(final_frame, rect, rect_resize, model, dimensions):
	'''Verify if rect is no overtaking sign. If so, putText on frame

	Args:
		final_frame (numpy.ndarray): image to draw the putText
		rect (numpy.ndarray): image rectangle (ROI) to verify if is no overtaking sign
		model: no overtaking model from train.py
		dimensions (tuple): width and heigh to resize the rect (ROI) image
	
	Returns:
		list: returns a list with the not no overtaking traffic signs 
	
	'''
	
	not_no_overtaking = []

	#HOG method
	H = hog(rect_resize, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=False, block_norm='L2-Hys')
	
	#predict the image based on model 
	pred = model.predict(H.reshape(1,-1))[0]
	
	if (pred.title()).lower() == 'pos':
		add_temp_coherence(True, "noOvertaking")
		#draw in video
		cv2.putText(final_frame,'Recognized: No overtaking ',(10,250), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
	else:
		not_no_overtaking = rect
		
	return not_no_overtaking


def predict_traffic_sign(circles, img, model, dimensions, final_frame, mask = 0):
	"""For each ROI found, verifies if it is or is not traffic sign

	Args:
		circles (numpy.ndarray): circles from Hough Circle method
		img (numpy.ndarray): image to cut the ROI 
		final_frame (numpy.ndarray): image to draw rectangles around the traffic signs
		model: traffic sinal model from train.py 
		dimensions (tuple): width and heigh to resize the ROI image

	Returns:
		tuple: returns 2 elements. The first one is a list of rectangles (ROIS) which are traffic sign
		The second one is a list of resized image of ROIs

	"""

	rects = []
	gray = []
	for i in circles[0,:]:
		x, y, radius = helpers.circle_values(i) 

		#DEBUG
		'''
		if frame_number > 749 and frame_number < 815: #placa de 80km
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
			mask_aux = mask.copy()
			mask = helpers.draw_circle(mask, (x,y), radius)
			mask = np.hstack((mask_aux, mask))
			cv2.imwrite("result/" + str(frame_number) + "-2mask-hough-" + str(contador) + ".jpg", mask)
		'''
		#Points to draw/take rectangle in image 
		x1_PRED, y1_PRED, x2_PRED, y2_PRED = helpers.rectangle_coord((x,y), radius, img.shape)
		
		#cut image
		rect = img[y1_PRED:y2_PRED, x1_PRED:x2_PRED].copy()  
						
		#For each ROI (rect) resize to dimension and verify if fits in model
		if rect.shape[0] > 0 and rect.shape[1] > 0: 
			img_resize = cv2.resize(rect, dimensions).copy()
		else:
			continue

		img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY).copy() #Put in grayscale

		#HOG method
		H = hog(img_gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=False, block_norm='L2-Hys')

		#predict the image based on model 
		pred = model.predict(H.reshape(1,-1))[0]
		if (pred.title().lower()) == "pos":
			#It is a traffic sign
			add_temp_coherence(True, None, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED))

			helpers.draw_circle (final_frame, (x,y), radius)
			cv2.rectangle(final_frame, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED), (0,0,255), 2)
			cv2.putText(final_frame,'Detected Traffic sign ',(10,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
			
			#It is a traffic sign
			rects.append(rect)
			gray.append(img_gray)
			
	
	return rects, gray

#################################################################################################################################
global contador
global frame_number
global temp_coherence #temporal coherence
global temp_image

#Argparse
ap = argparse.ArgumentParser()
#Add Arguments to argparse
ap.add_argument('-t', "--testing", required=True, help="path to the testing videos")
#ap.add_argument('-mf', "--modelFile", required=False, default='HOG_LinearSVC.dat', help='name of model file')
ap.add_argument('-d', "--dimension", required=False, default=(48,48), help="Dimension of training images (Width, Height)" )

args = vars(ap.parse_args())	#put in args all the arguments from argparse 

#Test class    
videos = glob.glob(args["testing"] + "*2*.mov")
dimensions = args["dimension"]

#Loading the models
print("Loading models...")
traffic_signs_model, no_overtaking_model, digits_model = load_models()

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #codec to save final video

directory = "../Final_Videos/"
if not os.path.exists(directory):
	os.makedirs(directory)	#create the directory if it does not exist

#Testing a model
for video in videos:
	#Predict for each video

	print("The current video is {}".format(video.split("/")[-1]))
	cap = cv2.VideoCapture(video)	#capture the video
	video_name = video.split("/")[-1].split(".")[0]	#Video name withour extensions
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
	#output_file = str(total_frames) + "\n" #Total number of frames for evaluate the model
	
	#To save a video with frames and rectangles in ROIs
	width_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
	video_out = cv2.VideoWriter(directory + video_name + '.avi', fourcc, frame_rate, (width_video, height_video))

	contador = 0
	temp_coherence = []
	temp_image = [] 
	for frame_number in range(0, total_frames):
		
		ret, frame = cap.read()	#capture frame-by-frame
		mask, img = preprocessing_frame(frame) #create a mask to HoughCircle
		
		temp_coherence.append([[frame_number, False, None, None, None]]) #list of list with five elements: frame_number, detected_sign, recognized_sign, coord1, coord2
		#temp_image.append([[frame_number, frame]])
		#DEBUG
		"""
		if frame_number > 749 and frame_number < 815: #placa de 80km
			maskk = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
			frame_clahe = np.hstack((frame, img, maskk))
			cv2.imwrite("result/"+ str(frame_number) + "-1frame-clahe-mask-" + str(contador) + ".jpg", frame_clahe)
		"""
		circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, minDist = 200, param1=50, param2=20, minRadius=5, maxRadius=150)
		
		#for each circle in a frame, return the rectangle image of a frame original that correspond a traffic sign 		
		if circles is not None:
			contador = contador + 1 #DEBUG
			circles = np.uint16(np.around(circles))
			rect, rect_resize = predict_traffic_sign(circles, img, traffic_signs_model, dimensions, frame, mask.copy()) #mask.copy for DEBUG
			
			list_not_no_overtaking = [] #list of images that is not no overtaking sign
			
			#Verify if it is no overtaking sign
			for (rectangle, roi_resize) in zip(rect, rect_resize):
				not_no_overtaking = predict_no_overtaking_sign(frame, rectangle, roi_resize, no_overtaking_model, dimensions) 		

				if not_no_overtaking != []:
					list_not_no_overtaking.append(not_no_overtaking)
			
			#if it is not no overtaking sign can be a speed limit sign
			for speed_limit in list_not_no_overtaking:
				contador = contador + 1 #DEBUG
				frame = predict_speed_limit_sign(frame, speed_limit, digits_model, (28,28))
		
		if len(temp_coherence) > 10:
			temp_coherence.pop(0)
		print(temp_coherence)
		print("\n")
		img = cv2.putText(frame, 'Frame: ' + str(frame_number),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)    
		video_out.write(img)

	cap.release() #Release the capture
	video_out.release() #Release the video writer
