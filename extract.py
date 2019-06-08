#IMPORT PACKAGES
import cv2
import glob

#Set Count to track no of frames and scale to resize the images
count = 0
scale = 0.75

#for loop to run over all the sample videos to extract images. We use the glob package to specify the path:
for videoPath in glob.glob("D:/DU SPRING 2019/1.CV/GROUP PROJECT/Video_Song_Actor_01/Actor_01/samples/*.mp4"): #Provide the path to your videos folder
    vidcap = cv2.VideoCapture(str(videoPath))
    success,image = vidcap.read()
    
    #Only if the frame is captured the command enters to loop to save the frame
    while success:
        #Extract the height and width to scale the image
        height, width, channel = image.shape
        
        #Calculating the new scaled height and width based on the scale factor
        new_height, new_width = scale*height, scale*width
        #new_height, new_width = 600, 1068
        
        #resize the captured frame using newly scaled heigth and width
        new_image = cv2.resize(image, (int(new_width), int(new_height)))
        #gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        
        new_image = cv2.GaussianBlur(new_image,(5,5),0)
        # save frame as JPEG file
        cv2.imwrite("frame%03d.jpg" % count, new_image)      
        
        #to read the next frame
        success,image = vidcap.read()
        
        #Keeping track of frame capture and its count
        print('Read a new frame and count: ', success, ' and ',count)
        count += 1