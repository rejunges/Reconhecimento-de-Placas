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

    #Remove a little noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    
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
    x, y = center
    prop_padding = int(radius/padding) #Proporcional padding
    
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

'''This function compute the area from a rectangle '''
def area (rect):
    width, heigth = rect.shape[1], rect.shape[0]
    
    return width * heigth 

'''This function compute the Jaccard similarity coefficient'''
def jaccardIndex(img_rect1, img_rect2):
    ri = img_rect1 & img_rect2
    ru = img_rect1 | img_rect2

    jaccardIndex = area(ri) / area(ru)
    return jaccardIndex
