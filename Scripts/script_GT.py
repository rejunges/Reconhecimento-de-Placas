"""
File name: script_GT.py
Description: This file rewrites the GT file, only mantaining the pictures with dimensions 40x40 or superior
Author: Renata Zottis Junges
Python Version: 3.6
"""

import glob

for file in glob.glob("*GT*.txt"): #Here put the name of ground truth file 
    print("File: " + file + "\nTo: " + file.split(".")[0])
    #Open the GT file 
    with open(file, 'r') as arq: 
        line = arq.readline() 
        #Open a file to write the new GT File
        with open(file.split(".")[0], "w") as fw:
             
            #Read line by line until the end of file
            while(line):
                frame, x, y, width, heigth, _ = line.split(",")
                if float(width) >= 40 and float(heigth) >= 40:
                    fw.write(line)
                line = arq.readline()
        fw.close() 
    arq.close() #Close file
