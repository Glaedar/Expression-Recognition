import os
import cv2
import numpy as np
import argparse
import glob

# File paths of the folders with the images in them
angerDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//training_data//Angry'
disgustedDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//training_data//Disgusted'
fearDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//training_data//Fear'
happyDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//training_data//Happy'
neutralDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//training_data//Neutral'
sadDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//training_data//Sad'
surprisedDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//training_data//Surprised'
testDir = 'C://Users//Andrew//Desktop//AI Project//Artificial Intelligence Coursework//Test_Data'


# function that can be used for the edge detection
def auto_canny(img, sigma = 0.33):
	v = np.median(img)
	
	lower = int(max (0, (1.0 - sigma) * v))
	upper = int(min (255, (1.0 - sigma) * v))
	edged = cv2.Canny(img,lower,upper)
	return edged
	

# rest of the functions pulls all the images for eachfolder, converts them to grey scale
# does a gaussian blur first then the edge detection and writes over the files in the folders

for imagePath in glob.glob(angerDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)
	
for imagePath in glob.glob(disgustedDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)
	
for imagePath in glob.glob(fearDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)
	
for imagePath in glob.glob(happyDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)
	
for imagePath in glob.glob(neutralDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)
	
for imagePath in glob.glob(sadDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)
	
for imagePath in glob.glob(surprisedDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)
	
	
for imagePath in glob.glob(testDir + "/*.jpg"):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3) , 0)
	edges = auto_canny(blurred)
	cv2.imwrite(imagePath, edges)