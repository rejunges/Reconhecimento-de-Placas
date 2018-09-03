'''
File name: change_lines_GT.py
Description: This file orders lines by frame in GT file.
Author: Renata Zottis Junges
Python Version: 3.6
'''

#python change_lines_GT.py -g new_gt.txt

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--groundtruth", required = True, help = "path t ground truth file")
args = vars(ap.parse_args())

filename_GT = args["groundtruth"]

#Open GT file
file_GT = open(filename_GT, "r")
lines_GT = file_GT.readlines()
new_list = []

file_output = open(filename_GT.split(".")[0] + "final.txt", "w")
for i in lines_GT:
	frame,x,y,w,h,code = i.split(',')
	new_list.append([int(float(frame)), x, y, w, h, code])

new_list.sort(key=lambda x: x[0])

for i in new_list:
	frame, x, y, w, h, code = i
	to_write = str(frame) + "," + x + "," + y + "," + w + "," + h + "," + code
	file_output.write(to_write)

