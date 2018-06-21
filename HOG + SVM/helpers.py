import numpy as np
import cv2
from skimage import exposure

'''This function saves original image and Hog Image side by side (Optional)'''
def print_hog(hogImage, name, img = None):
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0,255))
    hogImage = hogImage.astype("uint8")
    hogImage = cv2.cvtColor(hogImage, cv2.COLOR_GRAY2BGR)
    
    #Save in the paste Images-HOG
    if img is None:
        #Save only Hog Image the HOG image in the folder
        cv2.imwrite("Images-HOG/" + name, hogImage)
    else:
        #Save horizontal stack of original image and HOG image  
        result = np.hstack((img, hogImage))    
        cv2.imwrite("Images-HOG/" + name, result) 
    
'''This function  returns a mask (binary) containing the red color'''
def mask_hsv_red(hsv_img):
    #The red color in opencv has the hue values approximately in the range of 0 to 10 and 170 to 180 (opencv3 documentation)
    mask1 = cv2.inRange(hsv_img, np.array([0, 70, 50]), np.array([10, 255, 255]) )
    mask2 = cv2.inRange(hsv_img, np.array([170, 70, 50]), np.array([180, 255, 255]) )
    
    #Put together the two masks
    mask = cv2.bitwise_or(mask1, mask2)

    return mask

'''This function returns the coordinate x,y of center of circle and the radius'''
def circle_values(value):
    x = value[0]
    y = value[1]
    radius = value[2]
    return x, y, radius

'''This function returns a img with a draw green circle '''
def draw_circle (img, center, radius):
    #draw the outer circle
    cv2.circle(img, center, radius,(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img, center, 2, (0,0,255),3)
    
    return img

'''This function receives the center and radius of a circle, the padding and the shape of image. 
Return coordinates of two points to draw a rectangle''' 
def rectangle_coord (center, radius, padding, shape):
    x, y = np.int64(center)
    radius = np.int64(radius)
    prop_padding = np.int64(radius/padding) #Proporcional padding
    prop_padding = 0 #for jaccard image is better if the prop_padding was zero
    
    #Top left coordinates
    x1 = x - radius - prop_padding
    y1 = y - radius - prop_padding
    
    #Bottow rigth coordinates
    x2 = x + radius + prop_padding
    y2 = y + radius + prop_padding
    
    #Verify limits
    width, height = shape[1], shape[0]
    if (x2 > width): 
        x2 = width
    if (x1 < 0):
        x1 = 0
    if (y1 < 0):
        y1 = 0
    if (y2 > height):
        y2 = height
    
    return x1, y1, x2, y2


'''Receives two points from GT image and PRED image and calculate de iou'''
def intersection_over_union(point1_GT, point2_GT, point1_PRED, point2_PRED):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(point1_GT[0], point1_PRED[0])
	yA = max(point1_GT[1], point1_PRED[1])
	xB = min(point2_GT[0], point2_PRED[0])
	yB = min(point2_GT[1], point2_PRED[1])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (point2_GT[0] - point1_GT[0] + 1) * (point2_GT[1] - point1_GT[1] + 1)
	boxBArea = (point2_PRED[0] - point1_PRED[0] + 1) * (point2_PRED[1] - point1_PRED[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	return iou

'''Receives a RGB image and return a RGB image with Contrast Limited Adaptive Histogram Equalization (CLAHE) '''
def clahe (img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(6,6))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return bgr