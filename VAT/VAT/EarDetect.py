import numpy as np
import cv2

##Threshold pink colour for the ears
##Get the center of mass for the pink colour
##Draw a line for all the points
##Get tangents

cap = cv2.VideoCapture('C:/Users/Jack/Desktop/python/Project/IMG_0845.MOV')
cv2.namedWindow('image')

def nothing(x):
    pass

cv2.createTrackbar('r','image',0,255,nothing)
cv2.createTrackbar('g','image',0,255,nothing)
cv2.createTrackbar('b','image',0,255,nothing)

cv2.resizeWindow('image', 960, 540)

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('image', frame)

    #value_when_true if condition else value_when_false
    #threshold(image, fImage, 125, 255, cv::THRESH_BINARY);

    r = cv2.getTrackbarPos('r','image')
    rArr = cv2.inRange(frame[:,:,0],  0 if r <= 10 else (r-10), 255 if r >= 245 else (255-r))

    g = cv2.getTrackbarPos('g','image')
    gArr = cv2.inRange(frame[:,:,0],  0 if g <= 10 else (g-10), 255 if g >= 245 else (255-g))

    b = cv2.getTrackbarPos('b','image')
    bArr = cv2.inRange(frame[:,:,0],  0 if b <= 10 else (b-10), 255 if b >= 245 else (255-b))


    np.array(rArray)

    

    #difference between 255 and 0
    #

    print(e3)
    

    

    #reR, threshR = cv2.threshold(frame[:,:,0], cv2.getTrackbarPos('r','image') - 10, 255, cv2.THRESH_BINARY)
    #reG, threshG = cv2.threshold(frame[:,:,1], cv2.getTrackbarPos('g','image') - 10, 255, cv2.THRESH_BINARY)
    #reB, threshB = cv2.threshold(frame[:,:,2], cv2.getTrackbarPos('b','image') - 10, 255, cv2.THRESH_BINARY)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
