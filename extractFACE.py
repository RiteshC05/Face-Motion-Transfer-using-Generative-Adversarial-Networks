# CROP ONLY THE FACE AND EXTRACT THE LANDMARK POINTS

# Import necessary packages
import dlib
import glob
import cv2
import numpy as np
from PIL import Image

# Initialize the dlib face detector and the pre-trained shape predictor model that detects 68 landmarks on a detected face:
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Initialize the count to track the input images of a folder:
count = 0
k = 0
# To get the images inside a specified folder:
for image in glob.glob("D:/DU SPRING 2019/1.CV/GROUP PROJECT/Video_Song_Actor_01/Actor_01/newly/*.jpg"):
    
    # Read the image from the folder:
    img = cv2.imread(str(image));
    
    
    # Convert the image to grayscale to process using dlib:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Use the detector to detect the faces:
    faces = detector(gray)
    
    #Open a text file to save the input landmark points to a .txt file:
    file = open("frameslpts/frame%03d.txt" % count, "w")
    
    m = 0
    # Now detect landmarks for every faces identified in the faces rectangular object
    for face in faces:
        
        # Getting the bound box for the face found by the dlib detector
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        w = abs(x1-x2)
        h = abs(y1-y2)
        print(w,', ',h) # check this to change if condition
        if m==0:
            x11, y11, x22, y22 = x1, y1, x2, y2
            h1 = h
            w1=w
            m = 1
            
        croped = gray[y11:y11+h1, x11:x11+w1]
        croped_org = img[y11:y11+h1, x11:x11+w1]
        
        
        blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        # Use the pre-trained predictor model to detect the 68 landmark points
        landmarks = predictor(gray, face)
        #cv2.imshow("Cropped", croped_org)
        # To get all the 68 landmarks detected in the landmarks object:
        for i in range(0,68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            
            cv2.circle(blank_image,(x,y), 2, (255,255,255), -1) 
        
            #Write the x and y values to the file to the file
            file.write(f'{x},{y}\n')
        new_lands = blank_image[y11:y11+h1, x11:x11+w1]
        if w == 104 and h == 104:                   # To be commented out before finding maximum W and H, after finding that schange the values.
            cv2.imwrite("face%03d.jpg" % k, new_lands)  #Writing the landmarks image.
            cv2.imwrite("croppedface%03d.jpg" % k, croped_org) # writing the faces image.
        
    #write_name = 'im'+str(k)+'.jpg'
    #cv2.imwrite(write_name, img)
    k = k+1
    # Close the file we wrote
    file.close()
    
    # Increase the count for next variable:
    count = count+1
    
    key = cv2.waitKey(1)
    
    if key == 27:
        break

cv2.destroyAllWindows()