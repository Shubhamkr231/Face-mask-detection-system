import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
st.title("FACE MASK DETECTION SYSTEM")
st.sidebar.image("https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png")
choice=st.sidebar.selectbox("Menu",("HOME","URL","CAMERA"))
if(choice=="HOME"):
    st.image("https://tridentinfo.com/wp-content/uploads/2020/12/facemask.jpg")
elif(choice=="URL"):
    url=st.text_input("Enter your URL")
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1
        btn2=st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        vid=cv2.VideoCapture(url)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="nomask/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+1,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+1,y+w),(0,255,0),4)
                window.image(frame,channels='BGR')
elif(choice=="CAMERA"):
    cam=st.selectbox("Choose Camera",("None","Primary","Secondary"))
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1
        btn2=st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        if cam=="primary":
            cam=0
        else:
            cam=1
        vid=cv2.VideoCapture(cam)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="nomask/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+1,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+1,y+w),(0,255,0),4)
                window.image(frame,channels='BGR')
