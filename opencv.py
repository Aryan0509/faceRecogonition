import cv2
import numpy as np
cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
face_data=[]
dataset_path='data/'

file_name=input("enter name")
while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
 
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
  
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("video frame",frame)  

    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break


face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

  
np.save(dataset_path + file_name+'.npy',face_data)
print('data stored at '+dataset_path + file_name+'npy')

cap.release()  
cv2.destroyAllWindows()

 
