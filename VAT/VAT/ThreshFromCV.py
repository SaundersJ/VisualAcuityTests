#Ref: https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py
from __future__ import print_function
import cv2 as cv
import argparse

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
window_trackbar = 'Track Bar'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

## [low]
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_trackbar, low_H)
## [low]

## [high]
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_trackbar, high_H)
## [high]

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_trackbar, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_trackbar, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_trackbar, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_trackbar, high_V)

#parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
#parser.add_argument('--camera', help='Camera devide number.', default=0, type=int)
#args = parser.parse_args()

## [cap]
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
cap = cv.VideoCapture("C:/Users/Jack/Desktop/python/Project./IMG_0845.MOV")
#cap.set(CV_CAP_PROP_FRAME_WIDTH, cap.get(CV_CAP_PROP_FRAME_WIDTH)/2);
#cap.set(CV_CAP_PROP_FRAME_HEIGHT, cap.get(CV_CAP_PROP_FRAME_HEIGHT)/2);
## [cap]

## [window]
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.namedWindow(window_trackbar)

cv.resizeWindow(window_capture_name, 480, 270)
cv.resizeWindow(window_detection_name, 480, 270)  

## [window]

## [trackbar]
cv.createTrackbar(low_H_name, window_trackbar , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_trackbar , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_trackbar , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_trackbar , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_trackbar , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_trackbar , high_V, max_value, on_high_V_thresh_trackbar)

cv.setTrackbarPos(low_H_name, window_trackbar, 0)
cv.setTrackbarPos(high_H_name, window_trackbar, 86)
cv.setTrackbarPos(low_S_name, window_trackbar, 31)
cv.setTrackbarPos(high_S_name, window_trackbar, 255)
cv.setTrackbarPos(low_V_name, window_trackbar, 124)
cv.setTrackbarPos(high_V_name, window_trackbar, 229)

## [trackbar]

while True:
    ## [while]
    ret, frame = cap.read()
    if frame is None:
        break

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame, (low_H, low_S, low_V), (high_H, high_S, high_V))
    ## [while]

    ## [show]
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    ## [show]

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break