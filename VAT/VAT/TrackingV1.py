import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math

#==== User variables ====
fileName = "C:/Users/Jack/Desktop/python/Project/IMG_0845.MOV"
#

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

cap = cv.VideoCapture(fileName)

kalman = cv.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

previous = [[np.float32(100000)],[np.float32(1000000)]]

frameNumber = 0
ret, frame = cap.read()
height, width = frame.shape[:2]
centerWidth = width / 2
centerHeight = height / 2
fps = math.ceil(cap.get(cv.cv2.CAP_PROP_FPS))

file  = open("C:/Users/Jack/Desktop/python/Project/IMG_0845_GT.txt", "r")
lines = file.readlines() 
groundtruth = []
for l in lines:
    l = l.split("-")
    groundtruth.append([l[0], l[1].rstrip()])

groundtruth = [[int(i[0]), int(i[1])] for i in groundtruth]
file.close()

def length(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

