'''
This file is responsable for taking a path where exists the Training images
and create and save a model using HOG and Linear SVC (SVM). 
The default name of model file is HOG_LinearSVC.dat but can be set opcionally in argument. 

Command line example: python3 HOG_LinearSVC-training.py -t path_Training -mf name_model 
'''

#python HOG_LinearSVC-training.py -t ../../proibido_ultrapassar/neg_pos_48x48/ 
#python HOG_LinearSVC-training.py -t ../../60km_h/neg_pos_48x48/ -mf training_60km.dat
#python HOG_LinearSVC-training.py -t ../../80km_h/neg_pos_48x48/ -mf training_80km.dat

import numpy as np 
import pickle
import cv2
import argparse
from skimage.feature import hog
from imutils import paths
from sklearn.svm import LinearSVC
from helpers import print_hog

ap = argparse.ArgumentParser()
ap.add_argument('-t', "--training", required=True, help="path to the training images")
ap.add_argument('-mf', "--modelFile", required=False, default="HOG_LinearSVC.dat", help="name of model file")
args = vars(ap.parse_args())

data = []
labels = []

#Read all images in training directory
print("Computing HOG...")
for file in paths.list_images(args["training"]):
    file = file.replace('\\', '') #This treat if has some space in image name
    
    img_original = cv2.imread(file) #Image in grayscale
    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    #Hog method
    (H, hogImage) = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=True, block_norm='L2-Hys')
    
    #Call this function to print the HOG image
    #print_hog(hogImage, file.split('/')[-1], img_original)
    
    labels.append(file.split('/')[-2])
    data.append(H)

#Training a model
print("Training a Model with LinearSVC...")
model = LinearSVC()
model.fit(data,labels)

#Saving the model
modelFile = args['modelFile']
print("Saving the model in {}".format(modelFile))
list_pickle = open(modelFile, 'wb')
pickle.dump(model, list_pickle)
list_pickle.close()
