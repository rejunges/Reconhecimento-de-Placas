'''
This file is responsable for taking a model file and a path where exists the Testing videos
and find where is the images training in the model. 
The default name of model file is HOG_LinearSVC.dat but can be set opcionally in argument 
(should be correspondent to the model provided in training file). 

Command line example: python3 HOG_LinearSVC-testing.py -t path_Testing -mf name_model 
'''

#python HOG_LinearSVC-testing.py -t ../../vts/
#python HOG_LinearSVC-testing.py -t ../../vts/60km/ -mf training_60km.dat
#python HOG_LinearSVC-testing.py -t ../../vts/80km/ -mf training_80km.dat

import numpy as np 
import pickle
import cv2
import argparse
import glob
import time
from skimage.feature import hog
from sklearn.metrics import jaccard_similarity_score, confusion_matrix
from sklearn.svm import LinearSVC
from imutils import paths
from helpers import mask_hsv_red, circle_values, draw_circle, rectangle_coord, intersection_over_union, clahe

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
j = 0
width, height = args["dimension"] 


for file in glob.glob(args["testing"] + '*.MOV'): #Take all files (need to be video)
    print("The current video is {}".format(file.split('/')[-1]))
    cap = cv2.VideoCapture(file) #Capture the video
    file_name = file.split('/')[-1].split('.')[0] #File name without extensions 
    output_file = str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) + "\n" #Total number of frames for evaluate the model

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
        #img_msrcp = msrcp(img)
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
                cimg = draw_circle (cimg, (x,y), radius)

                #Points to draw/take rectangle in image 
                
                x1_PRED, y1_PRED, x2_PRED, y2_PRED = rectangle_coord((x,y), radius, padding, img.shape)
                
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
                    draw_circle (img, (x,y), radius)
                    cv2.rectangle(img, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED), (0,0,255), 2)
                    #Output for evaluation 
                    #output_file = output_file + "{} {} {} {} {}".format(frame_number, x1_PRED, y1_PRED, x2_PRED, y2_PRED) 
                        
                    #output_file = output_file + " 1\n"
                    img = cv2.putText(img,'Detectou: Proibido Ultrapassar', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)
                    #cv2.imwrite('PU_IMAGENS/Pos/' + str(frame_number) + "_" + str(j) + file_name + ".jpg", img_resize) #Write Positive samples
                #To write/save Negative samples uncomment the following lines
                #else:
                    #output_file = output_file + " 0\n"
                    #cv2.imwrite('PU_IMAGENS/Neg/' + str(frame_number) + "_" + str(j) + file_name + ".jpg", img_resize) #Write Negative samples

                #cv2.imwrite('Mask/mask' + str(j) +'.jpg', mask)   
                     
               
        img = cv2.putText(img,'Frame: ' + str(frame_number),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4)    
        video_out.write(img)
        
    print("Creating the predictions file for future evaluation")
    #Write the output
    '''
    file = open(file_name + "_PRED", "w")
    file.write(output_file)
    file.close() 
    '''
    cap.release() #Release the capture
    video_out.release() #Release the video writer
