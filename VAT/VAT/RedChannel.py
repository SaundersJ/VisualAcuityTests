import cv2 as cv
import numpy as np

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
window_trackbar = 'Track Bar'

cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.namedWindow(window_trackbar)

#cv.resizeWindow(window_capture_name, 480, 270)
#cv.resizeWindow(window_detection_name, 480, 270)  

def nothing(x):
    pass

cv.createTrackbar("Low_R", window_trackbar, 0, 255, nothing)
cv.createTrackbar("High_R", window_trackbar, 0, 255, nothing)
cv.createTrackbar("Low_G", window_trackbar, 0, 255, nothing)
cv.createTrackbar("High_G", window_trackbar, 0, 255, nothing)
cv.createTrackbar("Low_B", window_trackbar, 0, 255, nothing)
cv.createTrackbar("High_B", window_trackbar , 0, 255, nothing)

cv.setTrackbarPos("Low_R", window_trackbar, 80)
cv.setTrackbarPos("High_R", window_trackbar, 100)
cv.setTrackbarPos("Low_G", window_trackbar, 51)
cv.setTrackbarPos("High_G", window_trackbar, 93)
cv.setTrackbarPos("Low_B", window_trackbar, 110)
cv.setTrackbarPos("High_B", window_trackbar, 227)

fileName = "C:/Users/Jack/Desktop/python/Project/IMG_0848.MOV"

cap = cv.VideoCapture(fileName)
frameNumber = 0
while True:
    frameNumber = frameNumber + 1
    ## [while]
    ret, frame = cap.read()
    if frame is None:
        print("Break {}".format(frameNumber))
        break

    low_r = cv.getTrackbarPos("Low_R", window_trackbar)
    high_r = cv.getTrackbarPos("High_R", window_trackbar)
    low_g = cv.getTrackbarPos("Low_G", window_trackbar)
    high_g = cv.getTrackbarPos("High_G", window_trackbar)
    low_b = cv.getTrackbarPos("Low_B", window_trackbar)
    high_b = cv.getTrackbarPos("High_B", window_trackbar)

    #frameInv = cv.bitwise_not(frame)
    frameInv = 255 - frame
    frameInv = cv.cvtColor(frameInv ,cv.COLOR_BGR2HSV)

    #inRange(frameInv, ()

    #hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)


    #Mat1b mask1, mask2;
    #inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    #inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
    #Mat1b mask = mask1 | mask2;
    #imshow("Mask", mask);
    
    #	255	192	203

    #c = np.uint8([[[low_r, low_g, low_b]]])
    
    #hsv_c = cv2.cvtColor(c,cv2.COLOR_BGR2HSV)

    #(channel_h, channel_s, channel_v) = cv.split(hsv)

    #blur = cv.GaussianBlur(frameInv,(5,5),0)    #Any number of channels

    #ret,thr = cv.threshold(blur[:,:,2], low_r, high_r ,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #ret,thg = cv.threshold(blur[:,:,1], low_r, high_r ,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #ret,thb = cv.threshold(blur[:,:,0], low_g, high_g ,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    #ret,thb = cv.threshold(blur[:,:,0], low_b, high_b ,cv.THRESH_BINARY+cv.THRESH_OTSU)

    #zipped = np.dstack((thb, thg, thr))

    #ret,th = cv.threshold(blur, (low_r, low_g, low_b), (high_r, high_g, high_b) ,cv.THRESH_BINARY+cv.THRESH_OTSU)

    frame_threshold = cv.inRange(frameInv, (low_r, low_g, low_b), (high_r, high_g, high_b))
    #frame_threshold = cv.inRange(channel_v, low_r, high_r)
    mask = cv.erode(frame_threshold, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    
    #frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #blur = cv.GaussianBlur(frame,(5,5),0)
    #ret,th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    


    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, mask)

    

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

cap.release()

