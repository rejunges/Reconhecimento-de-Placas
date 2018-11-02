"""
File name: test.py
Description: This file identifies and recognizes traffic signs in videos
Author: Renata Zottis Junges
Python Version: 3.6
"""

#python test.py -t ../../vts/60km/ 
#python test.py -t ../../sinalizacao_vertical/ -c 1
#python test.py -t ../../Videos_RG/

import numpy as np 
import pickle
import cv2
import argparse
import glob
import time
import operator
import imutils
import os
import helpers
import time

from skimage.feature import hog
from sklearn import svm
from imutils import paths, contours
from scipy.spatial import distance

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
	""" Creates a binary mask to segments the digits from speed limit signs.

	Args:
		rect (numpy.ndarray): speed limit image from original frame (without resize)

	Returns:
		(numpy.ndarray): binary image from preprocessing
	"""

	#Take rect because rect has not been resized
	img_gray = cv2.cvtColor(rect.copy(), cv2.COLOR_BGR2GRAY)
#	img_gray = cv2.GaussianBlur(img_gray, (5,5), 0) #REUNIAO:Rever (3,3) ou nada

	# Threshold the image
	ret, img_gray = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV) #REUNIAO aumenta 90-testando (10 em 10) até 128+-

	return img_gray

def predict_speed_limit_sign(rect, model, dimensions, order_rs):
	"""Segments rect image in digits and checks the value (0-9) for each digit using the
	digits model. 

	Args:
		rect (numpy.ndarray): image rectangle (ROI) to verify which speed limit sign it is
		model: digits model from train.py
		dimensions (tuple): width and heigh to resize the digits image
		order_rs (int): used to organize the temporal coherence list
	"""

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

#This function is no overtaking + mandatory way
def predict_no_overtaking_sign(rect, rect_resize, model, dimensions, order_rs, roi_mask):
	"""Verify if rect is no overtaking sign

	Args:
		rect (numpy.ndarray): image rectangle (ROI) to verify if is no overtaking sign
		rect_resize (numpy.ndarray): resized rect from the predict traffic sign function
		model: no overtaking model from train.py
		dimensions (tuple): width and heigh to resize the rect (ROI) image
		order_rs (int): used to organize the temporal coherence list
	
	Returns:
		list: returns a list with the not no overtaking traffic signs 
	
	"""
	
	not_no_overtaking = []

	#HOG method
	H = hog(rect_resize, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=False, block_norm='L2-Hys')
	
	#predict the image based on model 
	pred = model.predict(H.reshape(1,-1))[0]
	
	if (pred.title()).lower() == 'pos':
		roi_width, roi_height = roi_mask.shape[1], roi_mask.shape[0]
		height_30percent, width_30percent = int((roi_height*30)/100), int((roi_width*30)/100)
		roi_mask = roi_mask[height_30percent:roi_height - height_30percent, width_30percent: roi_width - width_30percent].copy() #without borders
	
		if np.sum(roi_mask == 255) > 0:
			add_temp_coherence(True, "Proibido ultrapassar", order=order_rs)
		else:
			add_temp_coherence(True, "Passagem obrigatoria", order=order_rs)
		
	else:
		not_no_overtaking = rect
		
	return not_no_overtaking


def predict_traffic_sign(circles, img, model, dimensions, mask):
	"""For each ROI found, verifies if it is or is not traffic sign

	Args:
		circles (numpy.ndarray): circles from Hough Circle method
		img (numpy.ndarray): image to cut the ROI 
		model: traffic sinal model from train.py 
		dimensions (tuple): width and heigh to resize the ROI image
		mask (numpy.ndarray): mask from hough circle

	Returns:
		tuple: returns 3 elements. The first one is a list of rectangles (ROIS) which are traffic sign
		The second one is a list of resized image of ROIs. The third is list of mask ROI (without resizing)

	"""

	rects = []
	gray = []
	roi_masks = []
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
		roi_mask = mask[y1_PRED:y2_PRED, x1_PRED:x2_PRED].copy()

		#For each ROI (rect) resize to dimension and verify if fits in model
		if rect.shape[0] >= 37 and rect.shape[1] >= 37: #37x37 is the minimum size
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
			roi_masks.append(roi_mask)	

	return rects, gray, roi_masks


def modified_coherence():
	"""This function accomplishes the temporal coherence in the frames list  """
	if coherence_size <= 1:
		return 

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


	#To remove detected but does not exist
	flag_iou = False
	list_to_remove = []
	for last in temp_coherence[-1]:
		fn_last, ds_last, rs_last, c_last, r_last, m_last = last
		if c_last:
			x1_last, y1_last, x2_last, y2_last = helpers.rectangle_coord(c_last, r_last, frame.shape)
			for l_temp in temp_coherence[:-1]:
				#only computes if it was not modified
				for l in l_temp:
					fn, ds, rs, c, r, m = l
					if m == False and c:			
						x1, y1, x2, y2 = helpers.rectangle_coord(c, r, frame.shape)
						#calculate the intersection over union
						iou = helpers.intersection_over_union((x1_last, y1_last), (x2_last, y2_last), (x1, y1), (x2, y2))
						if iou > 0:
							flag_iou = True
							#continue to improve performance 
		if not flag_iou and ds_last:
			list_to_remove.append(last)
		flag_iou = False
	
	for l in list_to_remove:
		fn, ds, rs, c, r, m = l.copy()
		if ds == True:
			temp_coherence[-1].remove(l)
			temp_coherence[-1].append([fn, False, None, c, r, m])



	#Discovers length of frames lists
	length_dict = {}
	for l_temp in temp_coherence[:-1]:
		#only computes if it was not modified 
		cont = 0
		for l in l_temp:
			fn, ds, rs, c, r, m = l 
			if m == False:
				cont += 1
		if cont not in length_dict:
			length_dict[cont] = 1
		else:
			length_dict[cont] += 1

	#order dictionary by item 
	length_order = sorted(length_dict.items(), key = lambda kv: kv[1])
	biggest_length, number = length_order.pop()

	#at least N/2 frames have the same length then probably the new frame has too
	
	if number >= int(coherence_size/2):
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
				temp_coherence[-1].append([fn_last, True, rs, c, r, True])
			
			else: #copy the probably_signs
				while last_length < biggest_length and probably_signs:
					last_length += 1
					fn, ds, rs, c, r, m = traffic_sign_information_coherence(
						pos, probably_signs.pop(0))
					temp_coherence[-1].append([fn_last, True, rs, c, r, True])
					
		elif last_length == biggest_length:
			#Verifies if it has some None in rs 
			position_none = []
			n = 0
			for l in temp_coherence[-1]:
				fn_last, ds, rs, c, r, m = l
				if rs == None:
					position_none.append(n) #position where the rs is None
				n += 1
					
			if position_none: #rule 1: detected and not recognized
				probably_signs, pos = probably_signs_coherence(biggest_length)

				for l in temp_coherence[-1]:
					fn_last, ds_last, rs, c_last, r_last, m = l
					if rs in probably_signs:
						probably_signs.remove(rs)

				for p in position_none:
					least_distance = []
					fn_last, ds_last, rs_last, c_last, r_last, m_last = temp_coherence[-1][p]
					for frame_prob in temp_coherence[pos]: #pos from the probably_signs_coherence function
						fn, ds, rs, c, r, m = frame_prob
						if c != None and c_last != None: 
							least_distance.append([distance.euclidean(c_last, c), rs, c, r])
					#order least_distance
					if least_distance:
						least_distance.sort()
						dist, rs, c, r = least_distance.pop(0)
						if ds_last:
							temp_coherence[-1][p] = [fn_last, ds_last, rs, c_last, r_last, True]
						else:
							temp_coherence[-1][p] = [fn_last, True, rs, c, r, True]
					elif c_last == None and probably_signs:
						fn, ds, rs, c, r, m = traffic_sign_information_coherence(pos, probably_signs.pop(0))
						temp_coherence[-1][p] = [fn_last, True, rs, c, r, True]


def save_video():
	""" This function save the new frame (final frame with annotations) in the new video."""

	list_frame = temp_coherence[-1]
	number, final_frame = temp_image[-1]
	sign_count = 0
	valorH = 250

	for i in list_frame:
		fn, ds, rs, center, radius, modified = i
		if ds == True:
			x1_PRED, y1_PRED, x2_PRED, y2_PRED = helpers.rectangle_coord(center, radius, final_frame.shape)
			helpers.draw_circle (final_frame, center, radius)
			cv2.rectangle(final_frame, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED), (0,0,255), 2)
			#cv2.putText(final_frame,'Detected Traffic sign ', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
			cv2.putText(final_frame,'Detectou placa de transito', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
		if rs != None:
			valorH += sign_count*100 
			#cv2.putText(final_frame,'Recognized: ' + rs ,(10,valorH), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
			cv2.putText(final_frame,'Reconheceu: ' + rs ,(10,valorH), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
			sign_count += 1
		#only for metrics
		if  ds == False and rs == None:
			filename_output.write(str(frame_number) + ",0,0,0,0,False,18\n") 
		elif ds == True and rs == None:
			filename_output.write(str(frame_number) + "," + str(x1_PRED) + "," + str(y1_PRED) +"," + str(x2_PRED)
			 + "," +str(y2_PRED) + "," + "True,19\n")
		else:
			try:
				filename_output.write(str(frame_number) + "," + str(x1_PRED) + "," +
			                      str(y1_PRED) + "," + str(x2_PRED) + "," + str(y2_PRED) + "," + "True," + str(code_traffic[rs]) + "\n")
			except:
				filename_output.write(str(frame_number) + "," + str(x1_PRED) + "," +
                                    str(y1_PRED) + "," + str(x2_PRED) + "," + str(y2_PRED) + "," + "True,18\n")
	#img = cv2.putText(final_frame, 'Frame: ' + str(number), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)    
	img = cv2.putText(final_frame, 'Quadro: ' + str(number), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)    
	video_out.write(img)


def recognizing_signs(rect, rect_resize, dimensions, order_rs, roi_mask):
	"""This method calls others methods to discover which traffic sign was detected
	
	Args:
		rect (numpy.ndarray): image rectangle (ROI) to recognize sign
		rect_resize (numpy.ndarray): ROI resized
		dimensions (tuple): width and heigh to resize the rect (ROI) image
		order (int): used to organize the temporal coherence list
		roi_mask (numpy.ndarray): ROI in mask image
	"""

	not_no_overtaking = predict_no_overtaking_sign(rectangle, roi_resize, no_overtaking_model, dimensions, order_rs, roi_mask) 		

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
ap.add_argument('-c', "--coherence", required=False, default=1,
                help="Size of temporal coherence")

args = vars(ap.parse_args())	#put in args all the arguments from argparse 

#Test class    
videos = glob.glob(args["testing"] + "*2*.mov")
dimensions = args["dimension"]
coherence_size = int(args["coherence"])
if coherence_size < 1:
	coherence_size = 1

code_traffic = {0: "Proibido ultrapassar", 1: "10 km/h", 2: "20 km/h", 3: "30 km/h", 4: "40 km/h",
				5: "50 km/h", 6: "60 km/h", 7: "70 km/h", 8: "80 km/h", 9: "90 km/h", 10: "100 km/h", 11: "110 km/h",
				12: "120 km/h", 13: "Inicio de pista dupla", 14: "Fim de pista dupla", 15: "Passagem obrigatoria",
				16: "Parada obrigatoria", 17: "Intersecao em circulo", 18: "Erro", 19: "Detectou apenas"}
code_traffic = {y: x for x, y in code_traffic.items()}  # invert key and items

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
	
	#for metrics
	filename_output = open(video.split("/")[-1] + ".txt", "w+")
	filename_output.write(str(total_frames) + "\n")
	#To save a video with frames and rectangles in ROIs
	width_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
	video_out = cv2.VideoWriter(directory + video_name + '.avi', fourcc, frame_rate, (width_video, height_video))

	contador = 0
	temp_coherence = []
	temp_image = [] 
	final_time = 0
	for frame_number in range(0, total_frames):
		
		start_time = time.clock()
		ret, frame = cap.read()	#capture frame-by-frame
		mask, img = preprocessing_frame(frame) #create a mask to HoughCircle
		

		
		temp_coherence.append([[frame_number, False, None, None, None, False]]) #list of list with six elements: frame_number, detected_sign, recognized_sign, (x,y), radius, modified
		temp_image.append([frame_number, frame])
		#DEBUG
		"""
		if frame_number > 749 and frame_number < 815: #placa de 80km
			maskk = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
			frame_clahe = np.hstack((frame, img, maskk))
			cv2.imwrite("result/"+ str(frame_number) + "-1frame-clahe-mask-" + str(contador) + ".jpg", frame_clahe)
		"""
		circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, minDist = 50, param1=50, param2=20, minRadius=5, maxRadius=150)
		
		#for each circle in a frame, return the rectangle image of a frame original that correspond a traffic sign 		
		if circles is not None:
			contador = contador + 1 #DEBUG
			circles = np.uint16(np.around(circles))
			rect, rect_resize, roi_masks = predict_traffic_sign(circles, img, traffic_signs_model, dimensions, mask.copy()) #mask.copy for DEBUG
			
			list_not_no_overtaking = [] #list of images that is not no overtaking sign
			
			order_rs = 0 #used to temp_coherence
			#For each detected sign try to recognizing the traffic sign 
			for (rectangle, roi_resize, roi_mask) in zip(rect, rect_resize, roi_masks):
				recognizing_signs(rectangle, roi_resize, dimensions, order_rs, roi_mask)
				order_rs += 1
		
		if coherence_size == 1:
			save_video() 
			#print(temp_coherence)
			#print("\n")
			temp_coherence.pop(0)
			temp_image.pop(0)

		else:
			if len(temp_coherence) == coherence_size:
				modified_coherence()
				temp_coherence.pop(0)
				temp_image.pop(0)
			save_video()

		final_time = time.clock() - start_time + final_time
			#print(temp_coherence)
			#print(temp_image)
			#print("\n")

	cap.release() #Release the capture
	video_out.release() #Release the video writer
	filename_output.close()

print("Tempo de execução: {} segundos\nTotal de quadros: {}".format(final_time, total_frames))