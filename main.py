# Livestream video first
import numpy as np
import cv2

livestream = cv2.VideoCapture(0)

while(True):
    ret, frame = livestream.read()
    # print(ret)
    print(frame)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
livestream.release()
cv2.destroyAllWindows()
