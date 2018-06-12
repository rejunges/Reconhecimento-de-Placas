#python metrics.py
import glob
import argparse
from helpers import intersection_over_union
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

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
        print('Confusion matrix, without normalization')

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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#Put in gt_files the name of gt files
gt_files = glob.glob("*99*_GT")
isTrue = "Proibido ultrapassar"
isFalse = "Outros"


for filename_GT in gt_files:
    predictions = []
    ground_truth = []
    clip = filename_GT.split("_GT")[0]
    
    #Open GT file
    file_GT = open(filename_GT, "r")
    line_GT = file_GT.readline()
    
    #Open PRED FILE
    file_PRED = open("PRED_SP/" + clip + "_PRED", "r")
    total_frames = file_PRED.readline() #first line is the total number of frames
    total_frames = int(total_frames) 
    line_PRED = file_PRED.readline()

    print(total_frames)

    for frame in range(total_frames):
        if (line_PRED):
            #read line_PRED and save informations
            line_PRED = line_PRED.strip() 
            frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred = line_PRED.split()
            frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred = int(frame_PRED), int(x1_PRED), int(y1_PRED), int(x2_PRED), int(y2_PRED), int(pred)
            pred = isTrue if pred else isFalse
        
        if (line_GT):
            #Read line of GT file
            line_GT = line_GT.strip()     
            frame_GT, x, y, width_GT, heigth_GT = line_GT.split(",") #read all information of ground truth line
            frame_GT, x1_GT, y1_GT, width_GT, heigth_GT = int(frame_GT), int(float(x)), int(float(y)), int(float(width_GT)), int(float(heigth_GT)) #convert string to int
        
        #x1_GT and y1_GT are the top left coord, then the x2_GT e y2_GT are the bottom right coord
        x2_GT = x1_GT + width_GT
        y2_GT = y1_GT + heigth_GT 
        
        if (frame == frame_GT):
            ground_truth.append(isTrue)
            line_GT = file_GT.readline()

            if(frame == frame_PRED):
                #can be true positive or false negative
                x2_PRED = x1_PRED + width_GT
                y2_PRED = y1_PRED + heigth_GT

                #calculate the intersection over union
                iou =  intersection_over_union((x1_GT, y1_GT), (x2_GT, y2_GT), (x1_PRED, y1_PRED), (x2_PRED, y2_PRED))
                
                #Verify if pred is Positive and intersection over union > 0.5 
                if (pred == isTrue and iou > 0.5):
                    #true positive
                    predictions.append(pred)
                else:
                    #false negative
                    predictions.append(isFalse)

                #read the next line 
                line_PRED = file_PRED.readline()    
            elif(frame > frame_PRED):
                #Pred file ends
                predictions.append(isFalse)
            elif (frame < frame_PRED):
                predictions.append(isFalse)
            
        elif (frame < frame_GT):
            ground_truth.append(isFalse)
            if (frame == frame_PRED):
                predictions.append(pred)
                line_PRED = file_PRED.readline()
            elif (frame < frame_PRED):
                predictions.append(isFalse)
            elif (frame > frame_PRED):
                #Pred file ends
                predictions.append(isFalse)
 
        elif (frame > frame_GT):
            #ground truth file ends
            ground_truth.append(isFalse)
            if (frame == frame_PRED):
                predictions.append(pred)
                line_PRED = file_PRED.readline()    
            elif (frame < frame_PRED):
                predictions.append(isFalse)
            elif (frame > frame_PRED):
                #Pred file ends
                predictions.append(isFalse)
        
    #Close files to open others
    file_GT.close()
    file_PRED.close()

    class_names = [isTrue, isFalse]
    #ground_truth = np.invert(ground_truth)
    #predictions = np.invert (predictions)

    print("Confusion Matrix for {}".format(filename_GT))
    print(confusion_matrix(ground_truth, predictions))
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(ground_truth, predictions, labels = class_names)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                    title='Normalized confusion matrix')

    plt.show()
    print("Classification Report for {}".format(filename_GT))

    target_names = ['Positive', 'Negative']
    #print(classification_report(ground_truth, predictions, target_names=target_names))
    #print(ground_truth, len(predictions))
    