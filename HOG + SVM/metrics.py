#python metrics.py
import glob
import argparse
from helpers import intersection_over_union
from sklearn.metrics import classification_report, confusion_matrix

#Put in gt_files the name of gt files
gt_files = glob.glob("*_GT")

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

    for frame in range(total_frames):
        if (line_PRED):
            #read line_PRED and save informations
            line_PRED = line_PRED.strip() 
            frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred = line_PRED.split()
            frame_PRED, x1_PRED, y1_PRED, x2_PRED, y2_PRED, pred = int(frame_PRED), int(x1_PRED), int(y1_PRED), int(x2_PRED), int(y2_PRED), int(pred)
        
        if (line_GT):
            #Read line of GT file
            line_GT = line_GT.strip()     
            frame_GT, x, y, width_GT, heigth_GT = line_GT.split(",") #read all information of ground truth line
            frame_GT, x1_GT, y1_GT, width_GT, heigth_GT = int(frame_GT), int(float(x)), int(float(y)), int(float(width_GT)), int(float(heigth_GT)) #convert string to int
            
        #x1_GT and y1_GT are the top left coord, then the x2_GT e y2_GT are the bottom right coord
        x2_GT = x1_GT + width_GT
        y2_GT = y1_GT + heigth_GT 
        
        if (frame < frame_GT and frame < frame_PRED):
            #True negative
            ground_truth.append(0)
            predictions.append(0)
            
        elif (frame < frame_GT and frame == frame_PRED):
            #can be true negative or false positive
            ground_truth.append(0)
            predictions.append(pred)
            #read next line of line_PRED
            line_PRED = file_PRED.readline()
        
        elif (frame == frame_GT and frame == frame_PRED):
            #can be true positive or false negative
            x2_PRED = x1_PRED + width_GT
            y2_PRED = y1_PRED + heigth_GT

            #calculate the intersection over union
            iou =  intersection_over_union((x1_GT, y1_GT), (x2_GT, y2_GT), (x1_PRED, y1_PRED), (x2_PRED, y2_PRED))
            
            #Verify if pred is Positive and intersection over union > 0.5 
            if (pred and iou > 0.5):
                #true positive
                predictions.append(pred)
            else:
                #false negative
                predictions.append(0)
                
            ground_truth.append(1)

            #read the next line in both of files
            line_PRED = file_PRED.readline()
            line_GT = file_GT.readline()
        
        elif (frame == frame_GT and frame < frame_PRED):
            #false Negative
            ground_truth.append(1)
            predictions.append(0)       
            #Read next line in gt file
            line_GT = file_GT.readline()     
        elif (frame > frame_GT):
            #true negative or false positive
            ground_truth.append(0)
            if (frame == frame_PRED):
                predictions.append(pred)
                line_PRED = file_PRED.readline()
            elif (frame > frame_PRED):
                predictions.append(0)
        elif (frame > frame_PRED):
            predictions.append(0)
            if (frame == frame_GT):
                ground_truth.append(1)
            else:
                ground_truth.append(0)

    #Close files to open others
    file_GT.close()
    file_PRED.close()

    print("Confusion Matrix for {}".format(filename_GT))
    print(confusion_matrix(ground_truth, predictions))
    print("Classification Report for {}".format(filename_GT))

    target_names = ['Positive', 'Negative']
    print(classification_report(ground_truth, predictions, target_names=target_names))
    #print(ground_truth, len(predictions))
