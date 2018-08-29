'''
This file is responsable for create a confusion matrix and a classification report based on ground truth and predictions files.
Command line example: python3 metrics.py
'''

#python metrics_detection.py -p 2.mov.txt -g 2_gt.txt
#python metrics_detection.py -p pred_teste -g gt_teste
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
    ground_truth.append(isFalse)
    if (frame == frame_PRED):
        predictions.append(isTrue)
        line_PRED = file_PRED.readline()
    else:
        predictions.append(isFalse)

    return line_PRED


#Argparse
ap = argparse.ArgumentParser()
#Add Arguments to argparse
ap.add_argument('-p', "--predictions", required=True,
                help="path to predictions file")
ap.add_argument('-g', "--groundtruth", required=True,
                help="path to ground truth file")
args = vars(ap.parse_args())  # put in args all the arguments from argparse

filename_PRED, filename_GT = args["predictions"], args["groundtruth"]


predictions = []
ground_truth = []

#Open GT file
file_GT = open(filename_GT, "r")
line_GT = file_GT.readline()

total_lines = len(open(filename_PRED).readlines())

#Open PRED FILE
file_PRED = open(filename_PRED, "r")
total_frames = file_PRED.readline()  # first line is the total number of frames
total_frames = int(total_frames)
line_PRED = file_PRED.readline()

isTrue = "Detectou"
isFalse = "Nao detectou"

def read_PRED(line_PRED):
	line_PRED = line_PRED.strip()
	frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = line_PRED.split(
		",")
	frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = int(frame_PRED), int(
		x1_PRED), int(y1_PRED), int(x2_PRED), int(y2_PRED), bool(pred), int(code_PRED)

	return frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED

def read_GT(line_GT):
	line_GT = line_GT.strip()
	frame_GT, x, y, width_GT, heigth_GT, code_GT = line_GT.split(
		",")  # read all information of ground truth line
	frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT = int(frame_GT), int(float(x)), int(
		float(y)), int(float(width_GT)), int(float(heigth_GT)), int(code_GT)  # convert string to int

	return frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT


frame_PRED_ant, frame_PRED, frame_GT_ant, frame_GT = 0, 0, 0, 0

frame = 0
flag = True
for line in range(0, total_lines):
	if (line_PRED):
		#read line_PRED and save informations
		frame_PRED_ant = frame_PRED
		frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = read_PRED(line_PRED)
	
	if (line_GT):
		#Read line of GT file
		frame_GT_ant = frame_GT
		frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT = read_GT(line_GT)
	else:
		flag = False

	#x1_GT and y1_GT are the top left coord, then the x2_GT e y2_GT are the bottom right coord
	x2_GT = x1_GT + width_GT
	y2_GT = y1_GT + heigth_GT
#	print("Frame Pred", frame_PRED)
#	print("Frame GT", frame_GT)
	if frame_PRED >= frame_GT and flag:
		frame = frame_GT	
	else:
		frame = frame_PRED

	if (frame == frame_GT):
		ground_truth.append(isTrue)
		line_GT = file_GT.readline()

		if(frame == frame_PRED):
			#can be true positive or false negative
			x2_PRED = x1_PRED + width_GT
			y2_PRED = y1_PRED + heigth_GT

			#calculate the intersection over union
			iou = intersection_over_union(
				(x1_GT, y1_GT), (x2_GT, y2_GT), (x1_PRED, y1_PRED), (x2_PRED, y2_PRED))

			#Verify if pred is Positive and intersection over union > 0.5
			if (iou > 0.5):  # TODO: verify 0.5
				#true positive
				predictions.append(isTrue)
			else:
				#false negative
				predictions.append(isFalse)  # ERROR IOU

			#read the next line
			line_PRED = file_PRED.readline()
		else:
			predictions.append(isFalse)

	elif (frame < frame_GT):
		
		line_PRED = update_vectors(
			ground_truth, predictions, frame, frame_PRED, line_PRED, file_PRED, code_GT, code_PRED)

	elif (frame > frame_GT):
		#ground truth file ends
		line_PRED = update_vectors(
			ground_truth, predictions, frame, frame_PRED, line_PRED, file_PRED, code_GT, code_PRED)
	
	print("Frame: " + str(frame))

#print(predictions)
#print(ground_truth)
#print(total_frames)
#Close files to open others
file_GT.close()
file_PRED.close()

class_names = [isTrue, isFalse]

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

plt.savefig('confusion_matrix_detection.png', bbox_inches='tight')

'''
print("Classification Report for {}".format(filename_GT))

print(classification_report(ground_truth, predictions, target_names=class_names))
print(ground_truth, len(predictions))
'''
