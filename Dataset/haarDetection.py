# # plot photo with detected faces using opencv cascade classifier
import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
# # load the photograph
# pixels = cv2.imread('C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\Train\\frame100.jpg',)
# # cv2.imshow('pix',pixels)
# # load the pre-trained model
# classifier = CascadeClassifier('C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\haarcascade_frontalface_default.xml')
# # perform face detection
# bboxes = classifier.detectMultiScale(pixels)
# # print bounding box for each detected face
# for box in bboxes:
# 	# extract
# 	x, y, width, height = box
# 	x2, y2 = x + width, y + height
# 	# draw a rectangle over the pixels
# 	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# # show the image
# imshow('face detection', pixels)
# # keep the window open until we press a key
# waitKey(0)
# # close the window
# destroyAllWindows()

# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
# #function to detect face
# def detect_face (img):
# #convert the test image to gray image
# 	gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
# 	#load OpenCV face detector
# 	face_cas = cv2.CascadeClassifier ('C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\haarcascade_frontalface_default.xml')
# 	faces = face_cas.detectMultiScale (gray, scaleFactor=1.3, minNeighbors=4);
# 	#if no faces are detected then return image
# 	if (len (faces) == 0):
# 	return None, None
# 	#extract the face
# 	faces [0]=(x, y, w, h)
# 	#return only the face part
# 	return gray[y: y+w, x: x+h], faces [0]
# #this function will read all persons' training images, detect face #from each image
# #and will return two lists of exactly same size, one list
# def prepare_training_data(data_folder_path):
# #------STEP-1--------
# #get the directories (one directory for each subject) in data folder
# dirs = os.listdir(data_folder_path)
# faces = []
# labels = []
# for dir_name in dirs:
# #our subject directories start with letter 's' so
# #ignore any non-relevant directories if any
# if not dir_name.startswith("s"):
# continue;
# #------STEP-2--------
# #extract label number of subject from dir_name
# #format of dir name = slabel
# #, so removing letter 's' from dir_name will give us label
# label = int(dir_name.replace("s", ""))
# #build path of directory containin images for current subject subject
# #sample subject_dir_path = "training-data/s1"
# subject_dir_path = data_folder_path + "/" + dir_name
# #get the images names that are inside the given subject directory
# subject_images_names = os.listdir(subject_dir_path)
# #------STEP-3--------
# #go through each image name, read image,
# #detect face and add face to list of faces
# for image_name in subject_images_names:
# #ignore system files like .DS_Store
# if image_name.startswith("."):
# continue;
# #build image path
# #sample image path = training-data/s1/1.pgm
# image_path = 'C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\Train'
# #read image
# image = cv2.imread(image_path)
# #display an image window to show the image
# cv2.imshow("Training on image...", image)
# cv2.waitKey(100)
# #detect face
# face, rect = detect_face(image)
# #------STEP-4--------
# #we will ignore faces that are not detected
# if face is not None:
# #add face to list of faces
# faces.append(face)
# #add label for this face
# labels.append(label)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.destroyAllWindows()
# return faces, labels
# #let's first prepare our training data
# #data will be in two lists of same size
# #one list will contain all the faces
# #and other list will contain respective labels for each face
# print("Preparing data...")
# faces, labels = prepare_training_data("training-data")
# print("Data prepared")
# #print total faces and labels
# print("Total faces: ", len(faces))
# print("Total labels: ", len(labels))
# #create our LBPH face recognizer
# face_recognizer = cv2.face.createLBPHFaceRecognizer()
# #train our face recognizer of our training faces
# face_recognizer.train(faces, np.array(labels))
# #function to draw rectangle on image
# #according to given (x, y) coordinates and
# #given width and heigh
# def draw_rectangle(img, rect):
# (x, y, w, h) = rect
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# #function to draw text on give image starting from
# #passed (x, y) coordinates.
# def draw_text(img, text, x, y):
# cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
# #this function recognizes the person in image passed
# #and draws a rectangle around detected face with name of the subject
# def predict(test_img):
# #make a copy of the image as we don't want to chang original image
# img = test_img.copy()
# #detect face from the image
# face, rect = detect_face(img)
# #predict the image using our face recognizer
# label= face_recognizer.predict(face)
# #get name of respective label returned by face recognizer
# label_text = subjects[label]
# #draw a rectangle around face detected
# draw_rectangle(img, rect)
# #draw name of predicted person
# draw_text(img, label_text, rect[0], rect[1]-5)
# return img
# #load test images
# test_img1 = cv2.imread("C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\Test\\test0.jpg")
# test_img2 = cv2.imread("C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\Test\\test1.jpg")
# #perform a prediction
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
# print("Prediction complete")
# #create a figure of 2 plots (one for each test image)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# #display test image1 result
# ax1.imshow(cv2.cvtColor(predicted_img1, cv2.COLOR_BGR2RGB))
# #display test image2 result
# ax2.imshow(cv2.cvtColor(predicted_img2, cv2.COLOR_BGR2RGB))
# #display both images
# cv2.imshow("Tom cruise test", predicted_img1)
# cv2.imshow("Shahrukh Khan test", predicted_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.destroyAllWindows()

# import cv2
# import sys
# import os.path

# def detect(filename='C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection', cascade_file = "../lbpcascade_animeface.xml"):
#     if not os.path.isfile(cascade_file):
#         raise RuntimeError("%s: not found" % cascade_file)

#     cascade = cv2.CascadeClassifier('C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\lbpcascade_animeface.xml')
#     image = cv2.imread('C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\Anime-vs-Cartoon_1.jpg', cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
    
#     faces = cascade.detectMultiScale(gray,
#                                      # detector options
#                                      scaleFactor = 1.1,
#                                      minNeighbors = 5,
#                                      minSize = (24, 24))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

#     cv2.imshow("AnimeFaceDetect", image)
#     cv2.waitKey(0)
#     cv2.imwrite("out.png", image)

# # if len(sys.argv) != 2:
# #     sys.stderr.write("usage: detect.py <filename>\n")
# #     sys.exit(-1)
    
# detect()
import cv2
# from cv2 import cv
pixels = cv2.imread('C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\Train\\frame99.jpg',)
# cv2.imshow('pix',pixels)
# load the pre-trained model
classifier = CascadeClassifier('C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Haar\\Tom\\classifier\\cascade.xml')
# perform face detection
# bboxes = classifier.detectMultiScale(pixels)
faces = classifier.detectMultiScale(pixels, 
						scaleFactor=1.10, 
						minNeighbors=40, 
						minSize=(24, 24), 
						flags=cv2.CASCADE_SCALE_IMAGE
			)
# print bounding box for each detected face
for box in faces:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# show the image
imshow('face detection', pixels)
# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()


