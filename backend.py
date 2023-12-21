import cv2
import numpy as np
from keras.models import load_model
#To Read a video and detect mask on it and show it.
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5',compile=False)
vid=cv2.VideoCapture("http://192.168.0.113:8080/video")
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
        faces=facemodel.detectMultiScale(frame)
        for (x,y,l,h) in faces:
            #1.Cropping the face
            face_img=frame[y:y+h,x:x+l]
            #2.Resize the cropped Face
            face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
            #3.Conveting shape of image according to desired dimension of model
            face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
            #4.Normalize the image
            face_img=(face_img/127.5)-1
            p=maskmodel.predict(face_img)[0][0]
            if(p>0.9):
                cv2.rectangle(frame,(x,y),(x+l,y+h),(0,0,255),4)
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+h),(0,255,0),4)
        cv2.namedWindow("shubham window",cv2.WINDOW_NORMAL)
        cv2.imshow("shubham window",frame)
        key=cv2.waitKey(1)
        if(key==ord('x')):
            break
    else:
        break
cv2.destroyAllWindows()
