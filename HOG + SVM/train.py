"""
File name: train.py
Description: This file creates and saves a model using HOG and SVM. To use SVC you should especified
in -ml argument, otherwise the default is LinearSVM.

-ml 1 is used to LinearSVM (default)
-ml 2 is used to SVC 

Author: Renata Zottis Junges
Python Version: 3.6

"""

#python train.py -t ../../proibido_ultrapassar/neg_pos_48x48/ 
#python train.py -t ../../60km_h/neg_pos_48x48/ -mf training_60km.dat
#python train.py -t ../../80km_h/neg_pos_48x48/ -mf training_80km.dat
#python train.py -t ../../datasets/digits/ -mf digits.dat -ml 2

import numpy as np 
import pickle
import cv2
import argparse
import os
from skimage.feature import hog
from imutils import paths
from sklearn import svm
from helpers import print_hog

ap = argparse.ArgumentParser()
ap.add_argument('-t', "--training", required=True, help="path to the training images")
ap.add_argument('-mf', "--modelFile", required=True, help="name of model file")
ap.add_argument("-ml", "--machineLearning", required=False, default = 1, help="method of SVM (1-LinearSVM; 2- SVC)")
args = vars(ap.parse_args())

data = []
labels = []

#Read all images in training directory
print("Computing HOG...")
for file in paths.list_images(args["training"]):
    file = file.replace('\\', '') #This treat if has some space in image name
    
    img_original = cv2.imread(file) 
    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY) #Image in grayscale
    
    #Hog method
    (H, hogImage) = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise=True, block_norm='L2-Hys')

    #Call this function to print the HOG image
    #print_hog(hogImage, file.split('/')[-1], img_original)
    
    labels.append(file.split('/')[-2])
    data.append(H)

mode = args["machineLearning"]

#Training a model
if (int(mode) == 2):
    print("Training a Model with SVC...")
    #model = svm.SVC()
    model = svm.SVC(kernel='poly', degree=5)
else:
    #default is 1
    print("Training a Model with LinearSVC...")
    model = svm.LinearSVC()

model.fit(data,labels)
    
#Saving the model
directory = "../Models/"
modelFile = args['modelFile']
print("Saving the model in {}".format(directory + modelFile))

if not os.path.exists(directory):
    #create the directory if it does not exist
    os.makedirs(directory)

list_pickle = open(directory + modelFile, 'wb') #Save in the folder Models
pickle.dump(model, list_pickle)
list_pickle.close()
