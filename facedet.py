import cv2


imagePath = '.\people_face.png'
cascadeClassifierPath = 'C:\\python_codes\\PythonMachineLearning\\OLD_CODE\\FaceDetection\\haarcascade_frontalface_alt.xml'

cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
image = cv2.imread(imagePath)

#import numpy as np
#imgae = np.array(image, dtype=np.uint8)

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detectedFaces = cascadeClassifier.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

for(x,y, width, height) in detectedFaces:
	cv2.rectangle(image, (x, y), (x+width, y+height), (0,0,255), 10)
	
cv2.imwrite('after_face_detection.jpg', image)
