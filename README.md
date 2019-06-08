# CS583FinalProject

README
=======================================
This project maps the facial motions from a source video to a target video.

Training Phase- 

The training is done on the target video where the GAN learns to generate synthesized faces. the input to the GAN consists of each frames from the video sequences and the corresponsing landmark points from each frames.

For the acquiring the data, we first need to extract the frames from the the video sequence. Use the code extract.py
Copy all the video sequences that you wish your model to train on into a folder.
Change the folder path in the commented section in extract.py to the folder containing your video.

Syntax: 
python3 extract.py
--------------------------------------------------------------------------------------------------------
Extract Landmark points
Create two folders, one for landmarks images and the other for cropped images. Make the change as speciified in the comments in extractFACE.py to change folder path. 

Dependency Download : In order to extract the facial landmark points, you will need to 68-point landmark detector trained model. Use the link below to download the shape_predictor_68_face_landmarks.dat file from the link below.
https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

Run extractFACE.py to extract the faces and landmarks

Syntax:
python3 extractFACE.py
-----------------------------------------------------------------------------------------------------------
Training the Generative Adversarial Network
ganfinal.py - This code takes the input as the folders containing the cropped and landmarks.
Make the change to read the cropped image and the landmark points from your folder as specified in the comments in the code. 

This code uses the following libraries to run :
Tensorflow 9.1 ( Use GPU version to train the model faster)
Keras
OpenCV
PIL
dLib (http://dlib.net/)

Syntax:
python3 ganfinal.py
-----------------------------------------------------------------------------------------------------------

Temporal Smoothing --Optional 

You can run temp_smooth.py to smooth the video sequence in the temporal domain. This should be done before the training phase and after extracting the landmark points.

Syntax
python3 temp_smooth.py
---------------------------------------------------------------------------------------------------------
Additionally you can also use the cropped faces and landmarks provided in the Data For training zip file 
