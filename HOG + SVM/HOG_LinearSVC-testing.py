'''
This file is responsable for taking a model file and a path where exists the Testing videos
and find where is the images training in the model. 
The default name of model file is HOG_LinearSVC.dat but can be set opcionally in argument 
(should be correspondent to the model provided in training file). 

Command line example: python3 HOG_LinearSVC-testing.py -t path_Testing -mf name_model 
'''

#python HOG_LinearSVC-testing.py -t ../../vts/

import numpy as np 
import pickle
import cv2
import argparse
import glob
import time
from imutils import resize
from skimage.feature import hog
from imutils import paths, resize
from sklearn.svm import LinearSVC
from helpers import mask_hsv_red, circle_values, draw_circle, rectangle_coord, jaccardIndex

#Argparse
ap = argparse.ArgumentParser()
ap.add_argument('-t', "--testing", required=True, help="path to the testing videos")
ap.add_argument('-mf', "--modelFile", required=False, default='HOG_LinearSVC.dat', help='name of model file')
ap.add_argument('-d', "--dimension", required=False, default=(48,48), help="Dimension of training images (Width, Height)" )
args = vars(ap.parse_args())

#Loading the model
modelFile = args['modelFile']
print("Loading the model in {}".format(modelFile))
list_pickle = open(modelFile, 'rb')
model = pickle.load(list_pickle)
list_pickle.close()

#Test a model
print("Testing a Model")
padding = 3
j = 1
width, height = args["dimension"]

for file in glob.glob(args["testing"] + 'clip_i5s_0094*'): #Take all files (need to be video)
    print("The current video is {}".format(file.split('/')[-1]))
    cap = cv2.VideoCapture(file) #Capture the video
    
    #To save a video with frames and rectangles in ROIs
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #codec
    video_out = cv2.VideoWriter(file.split('/')[-1].split('.')[0] + '.avi', fourcc, 20.0, (1920, 1080))
    frame_number = -1

    #Open ground truth file 
    gt_file = open("clip_i5s_0094_GT", "r")
    line = gt_file.readline()

    while(cap.isOpened()):
        ret, frame = cap.read() #Capture frame-by-frame
        frame_number = frame_number + 1
        flag_gt = False
        #If the video has ended
        if not ret:
            break

        img = frame.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #mask with the red color of the image
        mask = mask_hsv_red(hsv)

        ##########################################
        #Ground Truth
        line = line.strip()
        frame_gt = line.split(",")[0]  
        if int(frame_gt) == frame_number:
            frame_gt, x, y, width_gt, heigth_gt = line.split(",")
            x, y, width_gt, heigth_gt = int(float(x)), int(float(y)), int(float(width_gt)), int(float(heigth_gt))
            #x and y are the top left coord
            x2 = x + width_gt
            y2 = y + heigth_gt
            #Take a rect of frame
            rect_gt = img[y:y2, x:x2]
            
            rect_gt = resize(rect_gt, width, height)
            if rect_gt.shape == (height, width, 3):
                flag_gt = True
            else:
                flag_gt = False
            
            #Read next gt
            line = gt_file.readline()

        ##########################################
        
        # HOUGH CIRCLE
        cimg = img.copy()
        #Blur mask to avoid false positives
        mask = cv2.medianBlur(mask, 5)

        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT,1, 50, param1=50, param2=20, minRadius=30, maxRadius=150)
        
        if circles is not None:
            #Exists at least one circle in a frame
            circles = np.uint16(np.around(circles))
            #For each circle found
            for i in circles[0,:]:
                x, y, radius = circle_values(i) 
                j = j + 1 #This line is necessary to write mask (optional)

                # draw the circle of ROI
                #cimg = draw_circle (cimg, (x,y), radius)

                #Points to draw/take rectangle in image 
                x1, y1, x2, y2 = rectangle_coord((x,y), radius, padding, img.shape)
               
                #cut image
                rect = img[y1:y2, x1:x2].copy()  

                #For each ROI (rect) resize to dimension and verify if fits in model
                try:
                    img_resize = resize(rect, width, height).copy()
                except:
                    continue
                #Put in grayscale
                img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                
                #HOG method
                (H, hogImage) = hog(img_gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=True, block_norm='L2-Hys')
                
                try:
                    #predict the image based on model 
                    pred = model.predict(H.reshape(1,-1))[0]
                except:
                    continue
                
                if flag_gt and img_resize.shape == (height, width, 3):
                    
                    print("JACCARD CALL")
                    print(jaccardIndex(rect_gt, img_resize))
                if (pred.title()).lower() == 'pos':
                    draw_circle (img, (x,y), radius)
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                        
                    #cv2.imwrite('PosResult/' + str(j) + ".jpg", img_resize) #Write Positive samples
                #To write/save Negative samples uncomment the following lines
                #else:
                #    cv2.imwrite('NegResult/' + str(j) + ".jpg", img_resize) #Write Negative samples

                #cv2.imwrite('Mask/mask' + str(j) +'.jpg', mask)        
        video_out.write(img)

    gt_file.close() #close ground truth file
    cap.release() #Release the capture
    video_out.release() #Release the video writer