import cv2
import sys
from face_detection import face
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt



model = load_model("model.h5")
result = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def printResult(predArray):
	total = (sum(t for t in predArray))
	for i in range(7):
		y = predArray[i] * 100 / total
		print(" " + '{:_<15}'.format(result[i]) + str(round(y,2))+"%")

def test(image):
    test_img =cv2.imread('test/' + image +'.png',0)
    img=cv2.resize(test_img,(48,48))
    img = img[:, :, np.newaxis]
    img=img.astype('float')/255.0
    img=np.expand_dims(img, axis=0)
    pred=model.predict(img)[0]
    printResult(pred)

    cv2.imshow('image',test_img)
    cv2.waitKey(1500)
    

def live():
	fd=face()
	cap=cv2.VideoCapture(0)
	ret=True
	while ret:
	    ret,frame=cap.read()
	    detected,x,y,w,h=fd.detectFace(frame)

	    if(detected is not None):
	        f=detected
	        detected=cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
	        detected=cv2.resize(detected,(48,48))
	        detected = detected[:, :, np.newaxis]
	        detected=detected.astype('float')/255.0
	        detected=np.expand_dims(detected,axis=0)
	        prediction=model.predict(detected)[0]

	        index=int(np.argmax(prediction))
	        print("Result: ", result[index])

	        #cv2.putText(frame, (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
	        cv2.rectangle(frame,(x,y),(x+w,y+h),(252,160,39),3)
	        cv2.imshow('onlyFace',f)
	    cv2.imshow('frame',frame)
	    if(cv2.waitKey(1) & 0XFF==ord('q')):
	        break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	if len(sys.argv) > 1:
		image = sys.argv[1]
		test(image)
	else:
		live()

