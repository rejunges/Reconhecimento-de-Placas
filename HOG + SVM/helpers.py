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

'''This function calculate the minimum point (x1,y1) and the maximum point (x2,y2) of the combination of
the ground truth points and predition image points. Return the minimum point and the width and height of a new image'''
def image_Jaccard_dimension (x1_GT, y1_GT, x2_GT, y2_GT, x1_PRED, y1_PRED, x2_PRED, y2_PRED):
    x1_img = x1_GT if (x1_GT < x1_PRED) else x1_PRED
    y1_img = y1_GT if (y1_GT < y1_PRED) else y1_PRED
    x2_img = x2_GT if (x2_GT > x2_PRED) else x2_PRED
    y2_img = y2_GT if (y2_GT > x2_PRED) else y2_PRED
    
    width_img = x2_img - x1_img
    height_img = y2_img - y1_img

    return x1_img, y1_img, width_img, height_img

'''This function return a two images(the ground truth and the predition image) based on the image_Jaccard_dimension function'''
def image_Jaccard (x1_GT, y1_GT, x2_GT, y2_GT, x1_PRED, y1_PRED, x2_PRED, y2_PRED):
    x1_img, y1_img, width_img, height_img = image_Jaccard_dimension(x1_GT, y1_GT, x2_GT, y2_GT, x1_PRED, y1_PRED, x2_PRED, y2_PRED)
    
    image_GT = np.zeros((height_img , width_img, 3), dtype = np.uint8)
    image_GT[:,:,1] = 255
    image_GT = cv2.cvtColor(image_GT, cv2.COLOR_BGR2HSV).copy()
    cv2.rectangle(image_GT, (x1_GT - x1_img, y1_GT - y1_img), (x2_GT - x1_img, y2_GT - y1_img), (150, 100, 100), cv2.FILLED)
    
    image_Pred = np.zeros((height_img , width_img, 3), dtype = np.uint8)
    image_Pred[:,:,0] = 255
    image_Pred[:,:,2] = 255
    image_Pred = cv2.cvtColor(image_Pred, cv2.COLOR_BGR2HSV).copy()
    cv2.rectangle(image_Pred, (x1_PRED - x1_img, y1_PRED - y1_img), (x2_PRED - x1_img, y2_PRED - y1_img), (150, 100, 100), cv2.FILLED)
    
    return image_GT, image_Pred

'''
def GT_contido_PRED (x1_GT, y1_GT, x2_GT, y2_GT, x1_PRED, y1_PRED, x2_PRED, y2_PRED):
    
    return (x1_GT >= x1_PRED and y1_GT >= y1_PRED and x2_GT <= x2_PRED and y2_GT <= y2_PRED)
'''