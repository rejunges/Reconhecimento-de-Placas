import numpy as np
import cv2
from skimage import exposure
import os

def clahe (img):
	""" Contrast Limited Adaptive Histogram Equalization (CLAHE) in BGR images
	
	This function changes the BGR image to LAB image and computes the CLAHE in the first LAB channel
	after that merge the channels and convert the LAB into BGR image again.

	Args:
		img (np.ndarray): BGR image
	
	Returns:
		img (np.ndarray): BGR image with CLAHE

	"""

	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	lab_planes = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(6,6))
	lab_planes[0] = clahe.apply(lab_planes[0])
	lab = cv2.merge(lab_planes)
	bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	
	return bgr


def print_hog(hogImage, name, img = None, folder = "../../Hog_Images/"):
	"""Saves HOG Image in a folder. Optionally saves original and HOG images side-by-side

	Args:
		hogImage (numpy.ndarray): HOG image provided by the HOG method  of skimage.feature
		name (str): file name to saves in folder
		img (numpy.ndarray): original image to saves side-by-side of HOG image (default = None)
		folder (str): path folder to saves the final image (HOG image or HOG and original image) (default = "../../Hog_Images/")

	"""

	hogImage = exposure.rescale_intensity(hogImage, out_range=(0,255))
	hogImage = hogImage.astype("uint8")
	hogImage = cv2.cvtColor(hogImage, cv2.COLOR_GRAY2BGR)

	if img is not None:  
		hogImage = np.hstack((img, hogImage)) #horizontal stack of original image and HOG image

	#Create the directory if it does not exist
	if not os.path.exists(folder):
		os.makedirs(folder)

	cv2.imwrite(folder + name, hogImage) #save image in folder


def mask_hsv_red(hsv_img):
	"""This function  create a mask (binary) containing the red color of a image in HSV space color

	Args:
		hsv_img (numpy.ndarray): image in HSV space color

	Returns:
		numpy.ndarray: binary mask of red color (white to represent red color and black to represent the others colors)

	"""

	#The red color in opencv has the hue values approximately in the range of 0 to 10 and 170 to 180 (opencv3 documentation)
	mask1 = cv2.inRange(hsv_img, np.array([0, 70, 50]), np.array([10, 255, 255]) )
	mask2 = cv2.inRange(hsv_img, np.array([170, 70, 50]), np.array([180, 255, 255]) )

	#Put together the two masks
	mask = cv2.bitwise_or(mask1, mask2)

	return mask


def circle_values(value):
	"""This function returns the coordinate x,y of center of circle and the radius

	Args:
		value (np.ndarray): array of coordinate x, y of a circle and the radius

	Returns:
		tuple: tuple with 3 elements, the coordinates x,y and the radius of circle

	"""
	x = value[0]
	y = value[1]
	radius = value[2]

	return x, y, radius


def draw_circle (img, center, radius):
	"""This function draws the center of circle (red color) and draw the outer circle (green color) 

	Args:
		img (np.ndarray): image to draw the circles
		center (tuple): the center coordinate (x, y) of circle
		radius (numpy.uint16): the circle radius
		
	Returns:
		numpy.ndarray: image with drawn circles (center and outer)

	"""

	#draw the outer circle
	cv2.circle(img, center, radius,(0,255,0),2)
	# draw the center of the circle
	cv2.circle(img, center, 2, (0,0,255),3)

	return img

def rectangle_coord (center, radius, shape, padding = 0):
	"""This function computes a bounding box around the circle. 
	Optionally the bounding box can have a padding.

	Args:
		center (tuple): the center coordinate (x, y) of circle
		radius (numpy.uint16): the circle radius
		shape (tuple): the shape of the image (height, width)
		padding (int): the padding to calculate de proporcional padding (radius/padding) to add in bounding box (default = 0)

	Returns:
		tuple: tuple with 4 elements, the top left coordinates x1,y1 and the bottow rigth coordinates x2,y2

	""" 

	x, y = np.int64(center)
	radius = np.int64(radius)

	if padding > 0:
		padding = np.int64(radius/padding) #Proporcional padding

	#Top left coordinates
	x1 = x - radius - padding
	y1 = y - radius - padding

	#Bottow rigth coordinates
	x2 = x + radius + padding
	y2 = y + radius + padding

	#Verify limits
	width, height = shape[1], shape[0]

	x2 = min(x2, width)
	x1 = max(0, x1)
	y1 = max(0, y1)
	y2 = min(y2, height)

	return x1, y1, x2, y2


def intersection_over_union(point1_GT, point2_GT, point1_PRED, point2_PRED):
	"""Calculates the intersection over union (jaccard coefficient) between the
	ground truth coordinates and predictions coordinates
	
	Args:
		point1_GT (tuple): coordinates x,y of ground truth point
		point2_GT (tuple): diagonal coordinates x,y of point1_GT 
		point1_PRED (tuple): coordinates x,y of prediction point
		point2_PRED (tuple): diagonal coordinates x,y of point1_PRED

	Returns:
		float: number between 0-1 representing the jaccard coefficient

	"""

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(point1_GT[0], point1_PRED[0])
	yA = max(point1_GT[1], point1_PRED[1])
	xB = min(point2_GT[0], point2_PRED[0])
	yB = min(point2_GT[1], point2_PRED[1])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (point2_GT[0] - point1_GT[0] + 1) * (point2_GT[1] - point1_GT[1] + 1)
	boxBArea = (point2_PRED[0] - point1_PRED[0] + 1) * (point2_PRED[1] - point1_PRED[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	return iou
