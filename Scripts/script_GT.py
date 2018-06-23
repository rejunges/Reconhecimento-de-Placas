'''This file is responsable for rewrite the GT file, only mantaining the pictures with dimensions 48x48 or superior'''
import glob

for file in glob.glob("*GT.txt"): #Here put the name of ground truth file 
    print("File: " + file + "\nTo: " + file.split(".")[0])
    #Open the GT file 
    with open(file, 'r') as arq: 
        line = arq.readline() 
        #Open a file to write the new GT File
        with open(file.split(".")[0], "w") as fw:
             
            #Read line by line until the end of file
            while(line):
                frame, x, y, width, heigth = line.split(",")
                if float(width) >= 48 and float(heigth) >= 48:
                    fw.write(line)
                line = arq.readline()
        fw.close() 
    arq.close() #Close file