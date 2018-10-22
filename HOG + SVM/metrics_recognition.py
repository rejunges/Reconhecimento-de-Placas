'''
File name: metrics_recognition.py
Description: This file create a confusion matrix based on recognition using prediction and ground truth files.
Author: Renata Zottis Junges
Python Version: 3.6
'''

#python metrics_recognition.py -p pred_teste -g gt_teste
#python metrics_recognition.py -p 2_pred.txt -g 2_GT
#python metrics_recognition.py -p 2_pred.txt -g 2_GT_wh

import glob
import argparse
from helpers import intersection_over_union
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os

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
        print("Matriz de confusão")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("Valor verdadeiro")
    plt.xlabel("Valor previsto")
    #plt.ylabel("Ground Truth label")
    #plt.xlabel("Predictions label")
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')

#This function update vectors(ground truth and predictions) when ground_truth is False (<frame or > frame)


def run_list_PRED(line_PRED, close):
	def read_PRED(line_PRED):
		line_PRED = line_PRED.strip()
		frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = line_PRED.split(
			",")
		frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = int(frame_PRED), int(
			x1_PRED), int(y1_PRED), int(x2_PRED), int(y2_PRED), bool(pred), int(code_PRED)

		return frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED

	list_PRED = []
	#Read Pred file and put information in  list_PRED
	frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = read_PRED(
		line_PRED)
	frame_PRED_ant = frame_PRED

	while frame_PRED == frame_PRED_ant and line_PRED:
		list_PRED.append([frame_PRED, x1_PRED, y1_PRED,
                    x2_PRED, y2_PRED, pred, code_PRED])

		#read line_PRED and save informations
		line_PRED = file_PRED.readline()
		if line_PRED:
			frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = read_PRED(
				line_PRED)
		else:
			close = True
	frame_PRED = frame_PRED_ant

	return list_PRED, frame_PRED, close, line_PRED


def run_list_GT(line_GT, flag):
	def read_GT(line_GT):
		line_GT = line_GT.strip()
		frame_GT, x, y, width_GT, heigth_GT, code_GT = line_GT.split(
			",")  # read all information of ground truth line
		frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT = int(frame_GT), int(float(x)), int(
			float(y)), int(float(width_GT)), int(float(heigth_GT)), int(code_GT)  # convert string to int

		return frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT

	list_GT = []
	frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT = read_GT(line_GT)
	frame_GT_ant = frame_GT

	while frame_GT == frame_GT_ant and line_GT:
		list_GT.append([frame_GT, x1_GT, y1_GT, width_GT,  heigth_GT, code_GT])
		line_GT = file_GT.readline()
		if line_GT:
			frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT = read_GT(line_GT)
		else:
			flag = False
	frame_GT = frame_GT_ant

	return list_GT, frame_GT, flag, line_GT


def frame_number(frame_PRED, frame_GT, flag):
	if frame_PRED >= frame_GT and flag:
		frame = frame_GT
	else:
		frame = frame_PRED

	return frame


def jaccard_index(x1_PRED, y1_PRED, width_GT, heigth_GT):
	x2_PRED = x1_PRED + width_GT
	y2_PRED = y1_PRED + heigth_GT
	x2_GT = x1_GT + width_GT
	y2_GT = y1_GT + heigth_GT
	#calculate the intersection over union
	iou = intersection_over_union(
		(x1_GT, y1_GT), (x2_GT, y2_GT), (x1_PRED, y1_PRED), (x2_PRED, y2_PRED))
	return iou


#Open GT file
file_GT = open(filename_GT, "r")
line_GT = file_GT.readline()

#Open PRED FILE
file_PRED = open(filename_PRED, "r")
total_frames = file_PRED.readline()  # first line is the total number of frames
total_frames = int(total_frames)
line_PRED = file_PRED.readline()

code_traffic = {0: "Proib. Ultrapassar", 1: "10 km/h", 2: "20 km/h", 3: "30 km/h", 4: "40 km/h",
                5: "50 km/h", 6: "60 km/h", 7: "70 km/h", 8: "80 km/h", 9: "90 km/h", 10: "100 km/h", 11: "110 km/h",
                12: "120 km/h", 13: "Inicio de pista dupla", 14: "Fim de pista dupla", 15: "Passagem obrigatoria",
                16: "Parada obrigatoria", 17: "Intersecao em circulo", 18: "Erro", 19: "Não reconheceu"}



frame_PRED_ant, frame_PRED, frame_GT_ant, frame_GT = 0, 0, 0, 0

frame = 0
flag = True
close = False
list_GT = []
list_PRED = []

while not close:
	if frame_PRED > frame_GT and flag:
		#run list_GT
		list_GT, frame_GT, flag, line_GT = run_list_GT(line_GT, flag)

	elif frame_PRED == frame_GT and flag:
		#Run both lists
		list_GT, frame_GT, flag, line_GT = run_list_GT(line_GT, flag)
		list_PRED, frame_PRED, close, line_PRED = run_list_PRED(line_PRED, close)

	else:
		#run List_PRED
		list_PRED, frame_PRED, close, line_PRED = run_list_PRED(line_PRED, close)

	#Order list by code_PRED and code_GT
	list_PRED.sort(key=lambda x: x[6])
	list_GT.sort(key=lambda x: x[5])
	frame = frame_number(frame_PRED, frame_GT, flag)
	#print("PRED: {}\nGT: {}\nFrame: {}\n\n ".format(list_PRED, list_GT, frame))

	#Here starts the list predictions and ground_truth
	if frame == frame_GT:
		if frame == frame_PRED:
			list_aux_PRED = list_PRED.copy()
			aux = 0
			for pred in list_aux_PRED:
				frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = pred
				list_aux_GT = list_GT.copy()
				flagIOU = True

				for gt in list_aux_GT:
					if flagIOU:
						frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT = gt

						iou = jaccard_index(x1_PRED, y1_PRED, width_GT, heigth_GT)
						if code_PRED != 18 and iou > 0.5:
							list_GT.remove([frame_GT, x1_GT, y1_GT, width_GT, heigth_GT, code_GT])
							predictions.append([code_PRED, frame])
							ground_truth.append([code_GT, frame])
							flagIOU = False

				if flagIOU:
					#print("{} {} {}".format(frame, code_GT, code_PRED))
					aux += 1

			list_GT = []
		else:
			list_aux_PRED = list_PRED.copy()
			for pred in list_aux_PRED:
				list_PRED.remove(pred)
				frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = pred
				while list_GT:
					list_GT.pop()

	elif (frame < frame_GT or frame > frame_GT):
		list_aux_PRED = list_PRED.copy()
		for pred in list_aux_PRED:
			list_PRED.remove(pred)
			frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred, code_PRED = pred


#print("Pred: {}\nGT: {}".format(predictions, ground_truth))
#print("Pred: {}\nGT: {}".format(len(predictions), len(ground_truth)))

#Close files to open others
file_GT.close()
file_PRED.close()

new_code_traffic = {}
for code in np.unique(predictions + ground_truth):
	sign = code_traffic.get(code)
	if sign != None:
		new_code_traffic[code] = sign

pred_aux = predictions.copy()
gt_aux = ground_truth.copy()
predictions = []
ground_truth = []

for pred, gt in zip(pred_aux, gt_aux):
	predictions.append(code_traffic.get(pred[0]))
	ground_truth.append(code_traffic.get(gt[0]))

codes_order = sorted(new_code_traffic.keys())
codes = []

for c in codes_order:
	codes.append(code_traffic[c])

class_names = codes

print("Confusion Matrix for {}".format(filename_GT))
# Compute confusion matrix

cnf_matrix = confusion_matrix(
	ground_truth, predictions, labels=class_names)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matriz de confusão')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

#plt.show()

plt.savefig('confusion_matrix_recognition.png', bbox_inches='tight')


print("Classification Report for {}".format(filename_GT))

print(classification_report(ground_truth, predictions, target_names=class_names))

print("Accuracy {0:.2f}".format(accuracy_score(ground_truth, predictions)))
