#Ref: https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py
from __future__ import print_function
import cv2 as cv
import argparse
from sklearn.cluster import MiniBatchKMeans
import math
import moviepy
from moviepy.video.io import VideoFileClip
from pathlib import Path
from datetime import datetime
import numpy as np
import ResizeVideo
import LogUtil
import math

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

fileName = "C:/Users/Jack/Desktop/python/Project/IMG_0848.MOV"
LogUtil.init(fileName)

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

def calculateLength(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

## [Resize Video]
LogUtil.write("Resizing Video")
resizedFileName = ResizeVideo.resizeVideo(fileName)
LogUtil.write("Finished Resizing Video")
##

## [cap]

cap = cv.VideoCapture(resizedFileName)
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
LogUtil.write("Starting Tracking")
frameNumber = 0

lengths = []

while True:
    frameNumber = frameNumber + 1
    ## [while]
    ret, frame = cap.read()
    if frame is None:
        break

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    data = []
    for i in range(len(frame_threshold)):
        for j in range(len(frame_threshold[i])):
            if frame_threshold[i][j] == 255:
                data.append([i, j])

    kMeans = MiniBatchKMeans(n_clusters=4)

    #calculate the lengths of the lines and add to the array

    try:
        kMeans.fit(data)
        centers = kMeans.cluster_centers_
        for c in centers:
            cv.circle(frame, (math.floor(c[1]), math.floor(c[0])), 20, (0,0,255), thickness=1)
            for oC in centers:
                cv.line(frame, (math.floor(c[1]), math.floor(c[0])), (math.floor(oC[1]), math.floor(oC[0])), (255,0,0), thickness=1)
                lengths.append(calculateLength(math.floor(c[1]), math.floor(c[0]), math.floor(oC[1]), math.floor(oC[0])))
    except ValueError:
        LogUtil.write("Frame Number {} : Too few datapoints for number of clusters".format(frameNumber))
        print("Frame Number {} : Too few datapoints for number of clusters".format(frameNumber))
    ## [while]

    ## [show]
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    ## [show]

    key = cv.waitKey(30)
    if frameNumber == 20 or key == ord('q') or key == 27:
        break
LogUtil.write("Finished Tracking")

#length, numberOfSimilarities
lenDict = {}

for l in lengths:
    found = False
    for key, value in lenDict.items():
        if abs(l - key) < 20:
            lenDict[key] = lenDict[key] + 1
            found = True
            break
    
    if not found:
        lenDict[l] = 1

lengthOfHead = 0
timesOfLength = 0
for key, value in lenDict.items():
    print("{} : {}".format(key, value))
    if key != 0.0 and value > timesOfLength:
        lengthOfHead = key
        timesOfLength = value

print("Chosing length {} with {} times".format(lengthOfHead, timesOfLength))
print("Showing clip again now only with lengths mathcing the length of the head")
cap.release()
## ====================================== ##

frameNo = 0
capFinal = cv.VideoCapture(resizedFileName)
gradientBefore = 0
while True:
    frameNo = frameNo + 1
    ret, frame = capFinal.read()
    if frame is None:
        break

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    data = []
    for i in range(len(frame_threshold)):
        for j in range(len(frame_threshold[i])):
            if frame_threshold[i][j] == 255:
                data.append([i, j])

    kMeans = MiniBatchKMeans(n_clusters=4)

    #calculate the lengths of the lines and add to the array
    gradients = [] 
    try:
        kMeans.fit(data)
        centers = kMeans.cluster_centers_
        for c in centers:
            cv.circle(frame, (math.floor(c[1]), math.floor(c[0])), 20, (0,0,255), thickness=1)
            for oC in centers:
                length = calculateLength(math.floor(c[1]), math.floor(c[0]), math.floor(oC[1]), math.floor(oC[0]))
                if length < (lengthOfHead + 10) and length > (lengthOfHead - 10):
                    cv.line(frame, (math.floor(c[1]), math.floor(c[0])), (math.floor(oC[1]), math.floor(oC[0])), (255,0,0), thickness=1)
                    gradients.append((math.floor(c[1]) - math.floor(oC[1]))/(math.floor(c[0]) - math.floor(oC[0])))
    except ValueError:
        LogUtil.write("Frame Number {} : Too few datapoints for number of clusters".format(frameNo))
        print("Frame Number {} : Too few datapoints for number of clusters".format(frameNo))
    
    averageGradient = np.mean(gradients)
    cv.putText(frame, "GradChange: {0:.2f}".format(abs(gradientBefore - averageGradient)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    gradientBefore = averageGradient
    ## [show]
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    ## [show]

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

LogUtil.write("Finished Program {}".format(datetime.now()))