'''
This file is responsable for taking a path where exists the Training images
and create and save a model using HOG and SVC. 
The default name of model file is digits.dat but can be set opcionally in argument. 

Command line example: python3 Digits_Training.py -t path_Training -mf name_model 
'''

#python Digits_Training.py -t ../../training_digits/ 

import numpy as np 
import pickle
import cv2
import argparse
from skimage.feature import hog
from imutils import paths
from sklearn import svm
from helpers import print_hog

ap = argparse.ArgumentParser()
ap.add_argument('-t', "--training", required=True, help="path to the training images")
ap.add_argument('-mf', "--modelFile", required=False, default="digits.dat", help="name of model file")
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
#Create a classifier: a support vector classifier
model = svm.SVC(gamma=0.001)
model.fit(data,labels)

#Saving the model
modelFile = args['modelFile']
print("Saving the model in {}".format(modelFile))
list_pickle = open(modelFile, 'wb')
pickle.dump(model, list_pickle)
list_pickle.close()
