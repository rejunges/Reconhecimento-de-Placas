import numpy as np
import cv2
from skimage import exposure

'''This function print grayscale image and Hog Image side by side (Optional)'''
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
    