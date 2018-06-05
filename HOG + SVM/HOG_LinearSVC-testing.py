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
from skimage.feature import hog
from sklearn.metrics import jaccard_similarity_score, confusion_matrix
from sklearn.svm import LinearSVC
from imutils import paths
from helpers import mask_hsv_red, circle_values, draw_circle, rectangle_coord, intersection_over_union

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

#Test a modelgt_file
print("Testing a Model")
padding = 3
j = 0
width, height = args["dimension"]

for file in glob.glob(args["testing"] + 'clip_i5s_0094*'): #Take all files (need to be video)
    print("The current video is {}".format(file.split('/')[-1]))
    cap = cv2.VideoCapture(file) #Capture the video
    file_name = file.split('/')[-1].split('.')[0] #File name without extensions 

    #To save a video with frames and rectangles in ROIs
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #codec
    video_out = cv2.VideoWriter(file_name + '.avi', fourcc, 20.0, (1920, 1080))
    frame_number = -1

    #Open ground truth file 
    file_GT = open(file_name, "r")
    line_GT = file_GT.readline()
    
    while(cap.isOpened()):
        ret, frame = cap.read() #Capture frame-by-frame
        frame_number = frame_number + 1
        flag_GT = False
        
        #If the video has ended
        if not ret:
            break

        img = frame.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Put img(frame) in grayscale
        mask = mask_hsv_red(hsv) #mask with the red color of the image

        #GROUND TRUTH
        if (line_GT):
            line_GT = line_GT.strip()
            frame_GT = line_GT.split(",")[0] 

            #if frame_GT is equal to frame_number then there is a traffic signal 
            if int(frame_GT) == frame_number:
                frame_GT, x, y, width_GT, heigth_GT = line_GT.split(",") #read all information of ground truth line
                x1_GT, y1_GT, width_GT, heigth_GT = int(float(x)), int(float(y)), int(float(width_GT)), int(float(heigth_GT)) #convert string to int
                
                #x1_GT and y1_GT are the top left coord, then the x2_GT e y2_GT are the bottom right coord
                x2_GT = x1_GT + width_GT
                y2_GT = y1_GT + heigth_GT 
                
                flag_GT = True #Flag true because there is a traffic signal in this frame
                line_GT = file_GT.readline() #Read next ground truth line
        
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
                j = j + 1 #This line is to save images in disk

                # draw the circle of ROI
                #cimg = draw_circle (cimg, (x,y), radius)

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
                
                try:
                    #predict the image based on model 
                    pred = model.predict(H.reshape(1,-1))[0]
                except:
                    continue

                if (pred.title()).lower() == 'pos':
                    if flag_GT:
                        #image_GT, image_Pred = image_Jaccard(x1_GT, y1_GT, x2_GT, y2_GT, x1_PRED, y1_PRED, x2_PRED, y2_PRED)
                        #JI = jaccard_similarity_score(image_GT.flatten(), image_Pred.flatten())
                        IOU = intersection_over_union((x1_GT, y1_GT), (x2_GT, y2_GT), (x1_PRED, y1_PRED), (x2_PRED, y2_PRED) )
                        
                        x2_PRED = x1_PRED + width_GT
                        y2_PRED = y1_PRED + heigth_GT
                        '''
                        image_GT = img[y1_GT:y2_GT, x1_GT:x2_GT].copy()
                        image_Pred = img[y1_PRED:y2_PRED, x1_PRED:x2_PRED].copy()
                        
                        
                        FINAL = np.hstack((image_GT, image_Pred))
                        cv2.imwrite('Video_rect/' + str(j) + "image_GT.jpg", image_GT)
                        cv2.imwrite('Video_rect/' + str(j) + "image_PRED.jpg", image_Pred)
                        cv2.imwrite('Video_rect/' + str(j) + "Final.jpg", FINAL)
                        '''
                        iou =  intersection_over_union((x1_GT, y1_GT), (x2_GT, y2_GT), (x1_PRED, y1_PRED), (x2_PRED, y2_PRED))
                        print(IOU, iou)
                    draw_circle (img, (x,y), radius)
                    cv2.rectangle(img, (x1_PRED,y1_PRED), (x2_PRED,y2_PRED), (0,0,255), 2)
                    #cv2.putText(img, "IoU: {:.2f}".format(iou*100), (x1_PRED, y1_PRED), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    #cv2.imwrite('Video_rect/Pos_SP/' + str(j) + file_name + ".jpg", img_resize) #Write Positive samples
                #To write/save Negative samples uncomment the following lines
                #else:
                    #cv2.imwrite('Video_rect/Neg_SP/' + str(j) + file_name + ".jpg", img_resize) #Write Negative samples

                #cv2.imwrite('Mask/mask' + str(j) +'.jpg', mask)        
        video_out.write(img)

    file_GT.close() #close ground truth file
    cap.release() #Release the capture
    video_out.release() #Release the video writer