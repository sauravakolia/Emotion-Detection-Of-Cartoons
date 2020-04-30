# import cv2
# img = cv2.imread("train/frame0.jpg")
# crop_img = img[0:498, 0:173]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)

# (left_x:  324   top_y:  101   width:  321   height:  229)
# # (left_x:  498   top_y:  173   width:  244   height:  198)
import os
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import pandas as pd 
import cv2


# print(len("C:/Users/Saurav Akolia/Desktop/u/hackerearth/EmotionDetection/Dataset/Test/"))

# im = Image.open(os.getcwd()+'\\'+"train\\frame100.jpg")
# width, height = im.size   # Get dimensions

# left = (width - 324)/2
# top = (height - 101)/2
# right = (width + 321)/2
# bottom = (height + 198)/2

# # Crop the center of the image
# im = im.crop((left, top, right, bottom))
# im.show()

# (left_x:  397   top_y:  212   width:  208   height:  150)
# im=cv2.imread("Train\\frame78.jpg")
# x1, y1, width, height =(108,4,252,191)
# x2, y2 = x1 + width, y1 + height
# crop_img = im[y1:y2, x1:x2]
# cv2.imwrite('73', crop_img)
# cv2.waitKey(0)

im=cv2.imread("C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\Train\\frame78.jpg")
x1, y1, width, height =(108,-4,252,191)
x2, y2 = x1 + width, y1 + height
crop_img = im[y1:y2, x1:x2]
cv2.imwrite("C:\\Users\\Saurav Akolia\\Desktop\\u\\hackerearth\\EmotionDetection\\Dataset\\CropedImages\\frame78.jpg", crop_img)
cv2.waitKey(0)

# df=pd.read_csv('testresult.csv')
# c=0
# for x in df.index:
# 	print(df['Location'][x])
# 	im = cv2.imread(df['Location'][x])
	
# 	# crop_img = img[y:y+h, x:x+w]
# 	# cv2.imshow("cropped", crop_img)
	
# 	# im=plt.imread(df['Location'][x])

# 	if(df['Tom'][x]):
# 		x1, y1, width, height =df['TL'][x],df['TT'][x],df['TW'][x],df['TH'][x]
# 		x2, y2 = x1 + width, y1 + height
# 		crop_img = im[y1:y2, x1:x2]
# 		cv2.imwrite("TestCropedImages\\"+df['Location'][x][75:], crop_img)
# 		cv2.waitKey(0)
# 		c=c+1

# 		# face_boundary = im[y1:y2,x1:x2]
# 		# face_image = Image.fromarray(face_boundary)
# 		# # face_image = face_image.resize(60,60)
# 		# face_array = asarray(face_image)

# 		# plt.xlim(0.5, 1.5)
# 		# plt.ylim(0.5,1.5)

		

# 		# plt.savefig(df['Location'][x][-10:])

# 	elif(df['Jerry'][x]):
# 		x1, y1, width, height =df['JL'][x],df['JT'][x],df['JW'][x],df['JH'][x]
# 		x2, y2 = x1 + width, y1 + height
# 		crop_img = im[y1:y2, x1:x2]
# 		# cv2.imshow("cropped", crop_img)
# 		cv2.imwrite("TestCropedImages\\"+df['Location'][x][75:],crop_img)
# 		cv2.waitKey(0)
# 		c=c+1
# 		# plt.savefig(df['Location'][x][-10:])

	
# 	else:
# 		cv2.imwrite("TestCropedImages\\"+df['Location'][x][75:],im)
		
# 		cv2.waitKey(0)
# 		c=c+1

# # 		# plt.savefig(df['Location'][x][-10:])
# # 	# x2, y2 = x1 + width, y1 + height
# # 	# face_boundary = im[y1:y2,x1:x2]
# # 	# face_image = Image.fromarray(face_boundary)
# # 	# # face_image = face_image.resize(60,60)
# 	# face_array = asarray(face_image)

# 	# plt.imshow(face_array)
# 	# plt.show()

# print(c)	