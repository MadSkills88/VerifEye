# Inspiration for more accurate eye tracking
# http://thume.ca/projects/2012/11/04/simple-accurate-eye-center-tracking-in-opencv/


# Livestream video first
import numpy as np
import cv2

livestream = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while(True):
    ret, frame = livestream.read()
    # print(ret)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Using Haar Cascades to Track Faces
    # https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html
    # print(frame)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Need to make eye detectino more accurate and then track pupils
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
livestream.release()
cv2.destroyAllWindows()
