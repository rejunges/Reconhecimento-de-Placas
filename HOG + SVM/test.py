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
import operator
from skimage.feature import hog
from sklearn import svm
from imutils import paths, contours
import imutils
from imutils import contours
import os
import helpers

def add_temp_coherence(detected_sign, recognized_sign, center=None, radius=None, order=None):
	""" Put new list in temp_coherence list 

	Args:
		detected_sign (bool): true if exists a sign in frame otherwise false
		recognized_sign (str): name of recognized sign otherwise None
		center (tuple): tuple with the center coordinate x and y of traffic sign circle
		radius (float): radius of traffic sign circle
		order (int): order of the recognized signal in the same frame (position in list)
	"""
	
	if order == None:
		#Predict traffic sign (first time)
		ant_temp = temp_coherence.pop() #remove the same frame informing 1 in detected sign
		fn, ds, rs, c, r, modified = ant_temp[-1] #take the last one
		
		if ds == False: #Before traffic sign identification
			atual = [fn, detected_sign, recognized_sign, center, radius, False]
			temp_coherence.append([atual])
		else:
			#more than one traffic sign in a frame, then need one more item in the list
			atual = [fn, detected_sign, recognized_sign, center, radius, False]	
			ant_temp.append(atual)
			temp_coherence.append(ant_temp)
	else:
		#Predict/recognizing traffic sign (with order in list)
		fn, ds, rs, c, r, m = temp_coherence[-1][order]
		temp_coherence[-1][order] = [fn, ds, recognized_sign, c, r, m] #change the position to a new recognized sign

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
	digits_model = open_model("digits.dat")	#third model is digits.dat

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

def predict_speed_limit_sign(rect, model, dimensions, order_rs):
	'''Segments rect image in digits and checks the value (0-9) for each digit using the
	digits model. 

	Args:
		rect (numpy.ndarray): image rectangle (ROI) to verify which speed limit sign it is
		model: digits model from train.py
		dimensions (tuple): width and heigh to resize the digits image
		order_rs (int): used to organize the temporal coherence list
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

		roi = cv2.resize(roi, dimensions) #resize to HOG 
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		ret, roi = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY_INV)
		#HOG method
		(H, hogImage) = hog(roi, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=True, block_norm='L2-Hys')
		
		#predict the image based on model 
		digits_pred = digits_model.predict(H.reshape(1,-1))[0]
		
		digit = (digits_pred.title()).lower()

		if digit == "1":
			digits = digits + "1"
			#continue reading because can be 10, 100, 120, 125
		elif len(digits) > 0: #If the first number is 1 then the need to read the others
			digits = digits + digit
		else: 
			add_temp_coherence(True, str(digit)+"0 km/h", order=order_rs)
			break

	if len(digits) > 0:
		add_temp_coherence(True, str(digits) + " km/h", order=order_rs)



def predict_no_overtaking_sign(rect, rect_resize, model, dimensions, order_rs):
	'''Verify if rect is no overtaking sign

	Args:
		rect (numpy.ndarray): image rectangle (ROI) to verify if is no overtaking sign
		model: no overtaking model from train.py
		dimensions (tuple): width and heigh to resize the rect (ROI) image
		order_rs (int): used to organize the temporal coherence list
	
	Returns:
		list: returns a list with the not no overtaking traffic signs 
	
	'''
	
	not_no_overtaking = []

	#HOG method
	H = hog(rect_resize, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=False, block_norm='L2-Hys')
	
	#predict the image based on model 
	pred = model.predict(H.reshape(1,-1))[0]
	
	if (pred.title()).lower() == 'pos':
		add_temp_coherence(True, "No overtaking", order=order_rs)
	else:
		not_no_overtaking = rect
		
	return not_no_overtaking


def predict_traffic_sign(circles, img, model, dimensions, mask = 0):
	"""For each ROI found, verifies if it is or is not traffic sign

	Args:
		circles (numpy.ndarray): circles from Hough Circle method
		img (numpy.ndarray): image to cut the ROI 
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
			add_temp_coherence(True, None, (x,y), radius)

			#It is a traffic sign
			rects.append(rect)
			gray.append(img_gray)
				
	return rects, gray

def probably_signs_coherence(biggest_length):
	
	probably_signs = []

	for i in range(len(temp_coherence[:-1])-1, -1, -1):  # read descending order
		if len(temp_coherence[i]) == biggest_length:
			for l in temp_coherence[i]:
				fn, ds, rs, c, r, m = l 
				probably_signs.append(rs)
		return probably_signs, i
	
	return probably_signs, -1 #Never occurs

def traffic_sign_information_coherence(position, traffic_sign):
	for l in temp_coherence[position]:
		fn, ds, rs, c, r, m = l
		if rs == traffic_sign:
			return l
	
	return []

def modified_coherence():
	"""Second method """
	
	#Discovers length of frames lists
	length_dict = {}
	for l_temp in temp_coherence[:-1]:
		#only computes if it was not modified or ds is False because rule 2 "Not detected and exists sign" 
		cont = 0
		for l in l_temp:
			fn, ds, rs, c, r, m = l 
			if m == False or ds == False:
				cont += 1
		if cont not in length_dict:
			length_dict[cont] = 1
		else:
			length_dict[cont] += 1

	#order dictionary by item 
	length_order = sorted(length_dict.items(), key = lambda kv: kv[1])
	biggest_length, number = length_order.pop()

	#at least 5 frames have the same length then probably the new frame has too
	if number >= 5:
		last_length = len(temp_coherence[-1])
		if last_length < biggest_length:
			probably_signs, pos = probably_signs_coherence(biggest_length)
			for l in temp_coherence[-1]:
				fn_last, ds, rs, c, r, m = l 
				if rs in probably_signs:
					probably_signs.remove(rs)
			# Now the len(probably_signs) == (biggest_length - last_length)
			if len(probably_signs) == 1: #only one sign, otherwise need to know the radius
				fn, ds, rs, c, r, m = traffic_sign_information_coherence(pos, probably_signs[0])
				temp_coherence[-1].append([fn_last, ds, rs, c, r, True])
			
			#else: #TODO: verify radius and probabilities
				

	"""
	if frame_number == 250:
		print(length_dict)
		print(biggest_length)
		print(number)
		print(temp_coherence)
	"""




"""
def modified_coherence():
	
	temp_dict = {}

	#create a dict with recognized sign and number of times it appears until the last item of temp_coherence 
	#Ex: {"60 km/h, No overtaking": 8, "60 km/h":1}
	for l_temp in temp_coherence[:-1]:
		comb_sign = ""
		for l in l_temp:
			fn, ds, rs, center, radius, modified = l
			if modified == False and rs != None: #ignore modified signs 
				comb_sign += rs + ","
		
		if comb_sign != "":
			comb_sign = comb_sign[:-1] #remove the last ","
			if comb_sign not in temp_dict:
				temp_dict[comb_sign] = 1
			else:
				temp_dict[comb_sign] += 1

	if len(temp_dict) == 0:
		return

	#order dictionary by item (number of times the combination sign appears)
	order_dict = sorted(temp_dict.items(), key=lambda kv: kv[1])
	probably_signs, _ = order_dict.pop()
	probably_signs = probably_signs.split(",") #probably signs receive a list with probably signs
	
	cont_ps = set() #cont probably signs
	for ps in probably_signs:
		cont_ps.add(ps)
	if frame_number == 249:
		print(temp_coherence)
		print(order_dict)
		print(probably_signs)
		print(cont_ps)

	l_temp = temp_coherence[-1] #list the last frame signs
	if len(probably_signs) == len(l_temp):
		#detected and did not recognized
		
		cont_rs = set()
		for l in l_temp:
			fn, ds, rs, center, radius, modified = l
			cont_rs.add(rs)	
		
		if cont_rs != cont_ps: # if equal then detected and did a correct recognizing
			pos = 0
			#detected and did a wrong or None recognizing	
			for l in l_temp:
				fn, ds, rs, center, radius, modified = l
				if ds == True and rs == None:
					#Detected and did not recognize
					if len(cont_ps) == 1: #if only exists one then probably should be this one
						sign = cont_ps.pop()
						temp_coherence[-1][pos] = [fn, ds, sign, center, radius, True]
						break
					#TODO: continuar aqui se o tamanho for maior que 1
				pos += 1
				#remove from probably signs the recognized sign
				
				#for ps in probably_signs:
				#	if rs == ps:
				#		probably_signs.remove(rs)
				


	#else:
		#verify 
	
	#n = 0
	#for l in l_temp:
	#	fn, ds, rs, center, radius, modified = l
	#	if modified == False:
	#		if ds == True and rs == None: #detected and did not recognize
	#			#Then choose which sign traffic is 
	#			#order dict by value 
	#			order_dict = sorted(temp_dict.items(), key=lambda kv: kv[1])
	#			probably_sign, _ = order_dict.pop()
	#			while probably_sign in cont_rs:
	#				if len(order_dict) > 0:
	#					probably_sign, _ = order_dict.pop()
	#				else:
	#					break #or continue?
	#			temp_coherence[-1][n] = [fn, ds, probably_sign, center, radius, True] #find the new value to recognized sign
	#		n += 1	
		
"""
def save_video():

	list_frame = temp_coherence[0]
	number, final_frame = temp_image[0]
	sign_count = 0
	valorH = 250

	for i in list_frame:
		fn, ds, rs, center, radius, modified = i
		if ds == True:
			x1_PRED, y1_PRED, x2_PRED, y2_PRED = helpers.rectangle_coord(center, radius, final_frame.shape)
			helpers.draw_circle (final_frame, center, radius)
			cv2.rectangle(final_frame, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED), (0,0,255), 2)
			cv2.putText(final_frame,'Detected Traffic sign ', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
		if rs != None:
			valorH += sign_count*100 
			cv2.putText(final_frame,'Recognized: ' + rs ,(10,valorH), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
			sign_count += 1
		
	img = cv2.putText(final_frame, 'Frame: ' + str(number), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)    
	video_out.write(img)

def recognizing_signs(rect, rect_resize, dimensions, order_rs):
	'''This method calls others methods to discover which traffic sign was detected
	
	Args:
		rect (numpy.ndarray): image rectangle (ROI) to recognize sign
		rect_resize (numpy.ndarray): ROI resized
		dimensions (tuple): width and heigh to resize the rect (ROI) image
		order (int): used to organize the temporal coherence list
	
	'''

	not_no_overtaking = predict_no_overtaking_sign(rectangle, roi_resize, no_overtaking_model, dimensions, order_rs) 		

	if not_no_overtaking != []: #if the ROI is not no overtaking
		#if it is not no overtaking sign can be a speed limit sign
		predict_speed_limit_sign(not_no_overtaking, digits_model, dimensions, order_rs)

	

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
		
		if len(temp_coherence) == 10:
			modified_coherence()

		temp_coherence.append([[frame_number, False, None, None, None, False]]) #list of list with six elements: frame_number, detected_sign, recognized_sign, (x,y), radius, modified
		temp_image.append([frame_number, frame])
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
			rect, rect_resize = predict_traffic_sign(circles, img, traffic_signs_model, dimensions, mask.copy()) #mask.copy for DEBUG
			
			list_not_no_overtaking = [] #list of images that is not no overtaking sign
			
			order_rs = 0 #used to temp_coherence
			#For each detected sign try to recognizing the traffic sign 
			for (rectangle, roi_resize) in zip(rect, rect_resize):
				recognizing_signs(rectangle, roi_resize, dimensions, order_rs)
				order_rs += 1
		
		"""
		#After temporal coherence (10 frames) 
		temp_dict = {}
		if frame_number > 10:
			#create a dict with recognized sign and number of times it appears
			for l_temp in temp_coherence:
				for l in l_temp:
					fn, ds, rs, center, radius = l
					if rs not in temp_dict:
						temp_dict[rs] = 0
					else:
						temp_dict[rs] += 1

			#Only modifies the third element in list based on others
			pos = 3
			l_temp = temp_coherence[pos]
			cont_rs = set() 
			for l in l_temp:
				fn, ds, rs, center, radius = l
				if rs != None:
					cont_rs.add(rs)
			
			n = 0
			for l in l_temp:
				fn, ds, rs, center, radius = l
				if ds == True and rs == None:
					#Then choose which sign traffic is
					#order dict by value 
					order_dict = sorted(temp_dict.items(), key=lambda kv: kv[1])
					probably_sign, _ = order_dict.pop()
					while probably_sign in cont_rs:
						if len(order_dict) > 0:
							probably_sign, _ = order_dict.pop()
						else:
							break #or continue?
					temp_coherence[pos][n] = [fn, ds, probably_sign, center, radius ] #find the new value to recognized sign
				n += 1	
			
			#Now save the frame in video
			save_video()
		"""
		save_video()
		if len(temp_coherence) > 10:
			temp_coherence.pop(0)
			temp_image.pop(0)
		
		#print(temp_coherence)
		#print(temp_image)
		#print("\n")

	
	"""
	while len(temp_coherence) > 0:
		save_video()
		temp_coherence.pop(0)
		temp_image.pop(0)
	"""
	cap.release() #Release the capture
	video_out.release() #Release the video writer
