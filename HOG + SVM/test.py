"""
File name: test.py
Description: This file identifies and recognizes traffic signs in videos
Author: Renata Zottis Junges
Python Version: 3.6
"""

#python test.py -t ../../vts/60km/ 

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
import os
import helpers

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
		tuple: returns a tuple with 3 elements, the traffic signal, no overtaking and digits models
	
	"""

	traffic_signals_model = open_model("trafficSigns.dat")  #first model is trafficSigns.dat
	no_overtaking_model = open_model("noOvertaking.dat")	#second model is noOvertaking.dat
	digits_model = open_model("digits.dat")	#third model is digits.dat

	return traffic_signals_model, no_overtaking_model, digits_model

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


def predict_no_overtaking_signal(final_frame, rect, rect_resize, model, dimensions):
	'''Verify if rect is no overtaking signal. If so, putText on frame

	Args:
		final_frame (numpy.ndarray): image to draw the put
		rect (numpy.ndarray): image rectangle (ROI) to verify if is no overtaking signal
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
		#draw in video
		cv2.putText(final_frame,'Recognized: No overtaking ',(10,250), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
	else:
		not_no_overtaking.append(rect)
		

	return not_no_overtaking


def predict_traffic_signal(circles, img, model, dimensions, final_frame):
	"""For each ROI found, verifies if it is or is not traffic sign

	Args:
		circles (numpy.ndarray): circles from Hough Circle method
		img (numpy.ndarray): image to cut the ROI 
		final_frame (numpy.ndarray): image to draw rectangles around the traffic signals
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
			#It is a traffic signal
			helpers.draw_circle (final_frame, (x,y), radius)
			cv2.rectangle(final_frame, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED), (0,0,255), 2)
			cv2.putText(final_frame,'Detected Traffic Signal ',(10,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
			
			#It is a traffic signal
			rects.append(rect)
			gray.append(img_gray)
			
	
	return rects, gray



#Argparse
ap = argparse.ArgumentParser()
#Add Arguments to argparse
ap.add_argument('-t', "--testing", required=True, help="path to the testing videos")
#ap.add_argument('-mf', "--modelFile", required=False, default='HOG_LinearSVC.dat', help='name of model file')
ap.add_argument('-d', "--dimension", required=False, default=(48,48), help="Dimension of training images (Width, Height)" )

args = vars(ap.parse_args())	#put in args all the arguments from argparse 

#Test class    
videos = glob.glob(args["testing"] + "*")
dimensions = args["dimension"]

#Loading the models
print("Loading models...")
traffic_signals_model, no_overtaking_model, digits_model = load_models()

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
	#output_file = str(total_frames) + "\n" #Total number of frames for evaluate the model
	
	#To save a video with frames and rectangles in ROIs
	video_out = cv2.VideoWriter(directory + video_name + '.avi', fourcc, 20.0, (1920, 1080))

	for frame_number in range(0, total_frames):
		ret, frame = cap.read()	#capture frame-by-frame
		mask, img = preprocessing_frame(frame) #create a mask to HoughCircle
		circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, minDist = 200, param1=50, param2=20, minRadius=5, maxRadius=150)

		#for each circle in a frame, return the rectangle image of a frame original that correspond a traffic sign 		
		if circles is not None:
			circles = np.uint16(np.around(circles))
			rect, rect_resize = predict_traffic_signal(circles, img, traffic_signals_model, dimensions, frame)
			
			not_not_overtaking = [] #list of images that is not no overtaking signal
			
			#Verify if it is no overtaking signal
			for (rectangle, roi_resize) in zip(rect, rect_resize):
				not_no_overtaking = predict_no_overtaking_signal(frame, rect, roi_resize, no_overtaking_model, dimensions) 		

			#if it is not no overtaking signal can be a speed limit signal
			
		img = cv2.putText(frame, 'Frame: ' + str(frame_number),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)    
		video_out.write(img)

	cap.release() #Release the capture
	video_out.release() #Release the video writer

'''
	print("The current video is {}".format(file.split('/')[-1]))
	cap = cv2.VideoCapture(file) #Capture the video
	file_name = file.split('/')[-1].split('.')[0] #File name without extensions 
	

	#To save a video with frames and rectangles in ROIs
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #codec
	video_out = cv2.VideoWriter(file_name + '.avi', fourcc, 20.0, (1920, 1080))
	frame_number = -1
	
	while(cap.isOpened()):
		ret, frame = cap.read() #Capture frame-by-frame
		frame_number = frame_number + 1
		
		#If the video has ended
		if not ret:
			break

		img = frame.copy()
		img = clahe(img)
		
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Put img(frame) in grayscale
		mask = mask_hsv_red(hsv) #mask with the red color of the image
	
			   
		# HOUGH CIRCLE
		cimg = img.copy()
		
		#Blur mask to avoid false positives
		mask = cv2.GaussianBlur(mask, (3,3), 0)
		circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, minDist = 200, param1=50, param2=20, minRadius=5, maxRadius=150)
		
		if circles is not None:
			#Exists at least one circle in a frame
			circles = np.uint16(np.around(circles))
			#For each circle found
			for i in circles[0,:]:
				x, y, radius = circle_values(i) 


				j = j + 1 #This line is to save images in disk

				# draw the circle of ROI
				#cimg = draw_circle (cimg, (x,y), radius)

				#Points to draw/take rectangle in image 
				
				x1_PRED, y1_PRED, x2_PRED, y2_PRED = rectangle_coord((x,y), radius, img.shape)
				
				#cut image
				rect = img[y1_PRED:y2_PRED, x1_PRED:x2_PRED].copy()  
								
				#For each ROI (rect) resize to dimension and verify if fits in model
				if rect.shape[0] > 0 and rect.shape[1] > 0: 
					img_resize = cv2.resize(rect, (width, height)).copy()
				else:
					continue
				
				#Put in grayscale
				img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY).copy()
				
				#HOG method
				(H, hogImage) = hog(img_gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=True, block_norm='L2-Hys')
				
				#predict the image based on model 
				pred = model.predict(H.reshape(1,-1))[0]
				
				if (pred.title()).lower() == 'pos':
					#It is a traffic signal
					draw_circle (img, (x,y), radius)
					cv2.rectangle(img, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED), (0,0,255), 2)
					
					img = cv2.putText(img,'Detectou: PLACA', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
					
					#Take rect because rect have not been resized
					img_gray = cv2.cvtColor(rect.copy(), cv2.COLOR_BGR2GRAY)
					img_gray = cv2.GaussianBlur(img_gray, (5,5), 0) 

					# Threshold the image
					ret, img_gray = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)

					#morphological operations to cleanup the thresholded image
					kernel = np.ones((5,5),np.uint8)
					opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
					
					# Find contours in the image
					cnts = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
					cnts = cnts[0] if imutils.is_cv2() else cnts[1]
					digitCnts = []
					
					img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
					#loop over the candidates of digit area 
					for c in cnts:
						#compute the bounding box 
						(x, y, w, h) = cv2.boundingRect(c)
						
						img_gray_w, img_gray_h = img_gray.shape[1], img_gray.shape[0]
						if w >= img_gray_w/4 and x > 3 and (x + w < img_gray_w - img_gray_w/10) and y > 3 and y < img_gray_h:
							#cv2.rectangle(img_gray, (x ,y), (x+w,y+h), (0,0,255), 2)
							digitCnts.append(c)

					#sort the contours from left-to-right
					if digitCnts:
						digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
				   
					digits = []
					img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
					#loop over each of digits:
					for c in digitCnts:
						(x, y, w, h) = cv2.boundingRect(c)
						roi = rect[y : y+h, x : x+w ] #extract the digit ROI

						roi = cv2.resize(roi, (28,28)) #resize to HOG 
						roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
						ret, roi = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY_INV)
						
						#HOG method
						(H, hogImage) = hog(roi, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=True, block_norm='L2-Hys')
						
						#predict the image based on model 
						digits_pred = digits_model.predict(H.reshape(1,-1))[0]
						if (digits_pred.title()).lower() == '0':
							cv2.imwrite('PLACA_MASK(DIA_2)/0/' + str(frame_number) + "_" + str(j) + file_name + ".jpg", roi) #Write Positive samples
						elif (digits_pred.title()).lower() == '6':
							cv2.imwrite('PLACA_MASK(DIA_2)/6/' + str(frame_number) + "_" + str(j) + file_name + ".jpg", roi) #Write Positive samples
						elif (digits_pred.title()).lower() == '8':
							cv2.imwrite('PLACA_MASK(DIA_2)/8/' + str(frame_number) + "_" + str(j) + file_name + ".jpg", roi) #Write Positive samples
						else:
							cv2.imwrite('PLACA_MASK(DIA_2)/outros/' + digits_pred.title() +  "_" + str(frame_number) + "_" + str(j) + file_name + ".jpg", roi) #Write Positive samples

						digits.append(digits_pred.title())
				#To write/save Negative samples uncomment the following lines
				#else:
					#output_file = output_file + " 0\n"
					#cv2.imwrite('PU_IMAGENS/Neg/' + str(frame_number) + "_" + str(j) + file_name + ".jpg", img_resize) #Write Negative samples

				#cv2.imwrite('Mask/mask' + str(j) +'.jpg', mask)   
					 
			   
		img = cv2.putText(img,'Frame: ' + str(frame_number),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)    
		video_out.write(img)
		
	cap.release() #Release the capture
	video_out.release() #Release the video writer
'''