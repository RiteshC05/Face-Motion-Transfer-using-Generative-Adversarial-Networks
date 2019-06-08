# Sowrirajan Sowmithran
# Ritesh Chidambaram
# CODE: TEMPORAL SMOOTHING - USING BI_DIRECTIONAL OPTICAL FLOW

import cv2
import glob
import numpy as np
import pandas as pd
import os
from tqdm import tqdm




imagePath = "D:/DU SPRING 2019/1.CV/GROUP PROJECT/Video_Song_Actor_01/Actor_01/framesIN/*.jpg"			#Enter the path to your images
landmarkPath = 'D:/DU SPRING 2019/1.CV/GROUP PROJECT/Video_Song_Actor_01/Actor_01/frameslpts'			#Enter the path to your landmark points

pyr = []

for images in glob.glob(imagePath):
    im = cv2.imread(str(images))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    pyr.append(gray.reshape(518400,))

pyr = np.array(pyr)		#Storing all images in a pyramid

lpts = []

#Finds all files in datafolder
filenames = os.listdir(landmarkPath)


for filename in tqdm(filenames):
    #Combines folder name and file name.
    
    path = os.path.join(landmarkPath,filename)
    pts = pd.read_csv(path,header=None,sep=',')
    pts=np.asarray(pts, dtype = 'float32')
    
    lpts.append(pts)
    

lpts = np.reshape(lpts,(504,136))
print(lpts)

print(np.asarray(lpts).shape)    
print(np.asarray(pyr).shape)

print("LOADING TEMP SMOOTHING")
flow_past = np.zeros((68,2), dtype = 'int');
for i in range(0,pyr.shape[0]):
    
    if i == 0:	#condition to determine flow of first frame
        new_pts = lpts[i].reshape(68,2);
        file = open("smoimg_pts/newlpts%03d.txt" % i, "w")
        for j in range(0,68):
            file.write(f'{new_pts[j,0]}, {new_pts[j,1]}\n')
        file.close()
    elif i == pyr.shape[0]-1 :	#condition to determine flow of last frame
        new_pts = lpts[i].reshape(68,2);
        file = open("smoimg_pts/newlpts%03d.txt" % i, "w")
        for j in range(0,68):
            file.write(f'{new_pts[j,0]}, {new_pts[j,1]}\n')
        file.close()
    else:			#condition to determine flow of remaning frames using window method
        flow_past, status1, err1 = cv2.calcOpticalFlowPyrLK(pyr[i-1].reshape(540,960), pyr[i].reshape(540,960), lpts[i-1].reshape(68,2), flow_past, maxLevel = 3, winSize = (21,21))
        flow_future, status2, err2 = cv2.calcOpticalFlowPyrLK(pyr[i].reshape(540,960), pyr[i+1].reshape(540,960), lpts[i-1].reshape(68,2), flow_past, maxLevel = 3, winSize = (21,21))
    
        new_pts = ((flow_past + lpts[i].reshape(68,2) + flow_future)/3)		#Average of past and future frame flows
        #new_pts = int(new_pts.reshape(136,))
        file = open("smoimg_pts/newlpts%03d.txt" % i, "w")
        for j in range(0,68):
            file.write(f'{new_pts[j,0]}, {new_pts[j,1]}\n')
        file.close()


new_lpts = []

temp_smoothed = 'D:/DU SPRING 2019/1.CV/GROUP PROJECT/Video_Song_Actor_01/Actor_01/smoimg_pts'
#Finds all files in datafolder
filenames = os.listdir(temp_smoothed)


for filename in tqdm(filenames):
    #Combines folder name and file name.
    
    path2 = os.path.join(temp_smoothed,filename)
    pts2 = pd.read_csv(path2,header=None,sep=',')
    pts2=np.asarray(pts2, dtype = 'float32')
    
    new_lpts.append(pts2)
    

new_lpts = np.reshape(new_lpts,(504,136))
print(new_lpts)

k = 0

for images in glob.glob(imagePath):
    im = cv2.imread(str(images))
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    point = new_lpts[k].reshape(68,2)
    #print(point)
    for l in range(0,68):
        x = point[l,0]
        y = point[l,1]
        cv2.circle(im,(x,y), 4, (255,0,0), -1) 
    
    write_name = 'im'+str(k)+'.jpg'
    cv2.imwrite(write_name, im)
    k = k+1
    #print(k)

cv2.destroyAllWindows()


        

