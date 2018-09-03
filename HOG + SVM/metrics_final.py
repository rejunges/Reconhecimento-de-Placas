'''
This file is responsable for create a confusion matrix and a classification report based on ground truth and predictions files.
Command line example: python3 metrics.py
'''

#python metrics_final.py -p 2.mov.txt -g 2_gt.txt
import glob
import argparse
from helpers import intersection_over_union
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        print("Matriz de confusão (sem normalização)")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("Ground Truth label")
    plt.xlabel("Predictions label")
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')

#This function update vectors(ground truth and predictions) when ground_truth is False (<frame or > frame)


def update_vectors(ground_truth, predictions, frame, frame_PRED, line_PRED, file_PRED, code_GT, code_PRED):
    ground_truth.append(18)
    if (frame == frame_PRED):
        predictions.append(code_PRED)
        line_PRED = file_PRED.readline()
    else:
        predictions.append(18)

    return line_PRED

#Argparse
ap = argparse.ArgumentParser()
#Add Arguments to argparse
ap.add_argument('-p', "--predictions", required=True, help="path to predictions file")
ap.add_argument('-g', "--groundtruth", required=True, help="path to ground truth file")
args = vars(ap.parse_args())  # put in args all the arguments from argparse

filename_PRED, filename_GT = args["predictions"], args["groundtruth"]

code_traffic = {0: "Proibido Ultrapassar", 1: "10 km/h", 2: "20 km/h", 3: "30 km/h", 4:"40 km/h",
                5: "50 km/h", 6: "60 km/h", 7: "70 km/h", 8: "80 km/h", 9: "90 km/h", 10: "100 km/h", 11: "110 km/h",
				12: "120 km/h", 13: "Inicio de pista dupla", 14: "Fim de pista dupla", 15: "Passagem obrigatoria",
				16: "Parada obrigatoria", 17: "Intersecao em circulo", 18: "Erro", 19:"Detectou apenas"}

predictions = []
ground_truth = []

#Open GT file
file_GT = open(filename_GT, "r")
line_GT = file_GT.readline()

#Open PRED FILE
file_PRED = open(filename_PRED, "r")
total_frames = file_PRED.readline()  # first line is the total number of frames
total_frames = int(total_frames)
line_PRED = file_PRED.readline()

for frame in range(total_frames):
	if (line_PRED):
		#read line_PRED and save informations
		line_PRED = line_PRED.strip()
		frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = line_PRED.split(",")
		frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = int(frame_PRED), int(
			x1_PRED), int(y1_PRED), int(x2_PRED), int(y2_PRED), bool(pred), int(code_PRED)
		
	if (line_GT):
		#Read line of GT file
		line_GT = line_GT.strip()
		frame_GT, x, y, width_GT, heigth_GT, code_GT = line_GT.split(",")  # read all information of ground truth line
		frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT = int(frame_GT), int(float(x)), int(
			float(y)), int(float(width_GT)), int(float(heigth_GT)), int(code_GT)  # convert string to int

	#x1_GT and y1_GT are the top left coord, then the x2_GT e y2_GT are the bottom right coord
	x2_GT = x1_GT + width_GT
	y2_GT = y1_GT + heigth_GT

	if (frame == frame_GT):
		ground_truth.append(code_GT)
		line_GT = file_GT.readline()

		if(frame == frame_PRED and pred == True):
			#can be true positive or false negative
			x2_PRED = x1_PRED + width_GT
			y2_PRED = y1_PRED + heigth_GT

			#calculate the intersection over union
			iou = intersection_over_union(
				(x1_GT, y1_GT), (x2_GT, y2_GT), (x1_PRED, y1_PRED), (x2_PRED, y2_PRED))

			#Verify if pred is Positive and intersection over union > 0.5
			if (iou > 0.5): #TODO: verify 0.5 
				#true positive
				predictions.append(code_PRED)
			else:
				#false negative
				predictions.append(18) #ERROR IOU

			#read the next line
			line_PRED = file_PRED.readline()
		else:
			predictions.append(18)

	elif (frame < frame_GT):
		line_PRED = update_vectors(
			ground_truth, predictions, frame, frame_PRED, line_PRED, file_PRED, code_GT, code_PRED)

	elif (frame > frame_GT):
		#ground truth file ends
		line_PRED = update_vectors(
			ground_truth, predictions, frame, frame_PRED, line_PRED, file_PRED, code_GT, code_PRED)

#Close files to open others
file_GT.close()
file_PRED.close()
"""
codes_order = sorted(code_traffic.keys())
codes = []
for c in codes_order:
	
	codes.append(code_traffic[c])

print(codes)
class_names = codes

#print(len(ground_truth))
#print(len(predictions))
print("Confusion Matrix for {}".format(filename_GT))
# Compute confusion matrix
cnf_matrix = confusion_matrix(
	ground_truth, predictions, labels=class_names)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
						title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

#plt.show()

plt.savefig('confusion_matrix.png', bbox_inches='tight')
"""
'''
print("Classification Report for {}".format(filename_GT))

print(classification_report(ground_truth, predictions, target_names=class_names))
print(ground_truth, len(predictions))
'''
