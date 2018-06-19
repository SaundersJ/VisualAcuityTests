import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math
from scipy.cluster.vq import kmeans,vq
import KalmanManager

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

fileName = "C:/Users/Jack/Desktop/python/Project/IMG_0849.MOV"

cap = cv.VideoCapture(fileName)

KalmanManager.initKalman([[1,1],[2,2]])

#kalman = cv.KalmanFilter(4,2)
#kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
#kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
#kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

previous = [100000, 100000]

meas = []   #list
pred = []   #list
mp = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction

frameNumber = 0
centerLengths = []

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

def getlength(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

while True:
    frameNumber = frameNumber + 1
    ret, frame = cap.read()

    if frame is None:
        print("Break {}".format(frameNumber))
        break

    mask = frame
    mask = 255 - mask
    mask = cv.cvtColor(mask ,cv.COLOR_BGR2HSV)

    maskH = mask[:,:,0]
    maskS = mask[:,:,1]
    maskV = mask[:,:,2]

    ret, th1 = cv.threshold(maskH, 0, 255 ,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, th2 = cv.threshold(maskS, 0, 255 ,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, th3 = cv.threshold(maskV, 0, 255 ,cv.THRESH_BINARY+cv.THRESH_OTSU)

    mask = np.dstack((th1,th2,th3))

    mask = cv.cvtColor(mask ,cv.COLOR_HSV2BGR)
   
    mask[mask == 255] = 0

    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask[mask != 0] = 255
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    #data = np.argwhere(mask == 255)

    #mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    im2, cnts, hierarchy = cv.findContours(mask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:3]

    centers = []

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    #cN = 0
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv.boundingRect(approx)
        if h >= 15:
            # if height is enough
            # create rectangle for bounding
            rect = (x, y, w, h)
            #rects.append(rect)
            cv.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 1);
            centerX = math.floor(x + (w/2))
            centerY = math.floor(y + (h/2))
            cv.circle(mask, (centerX, centerY), 10, (0,0,255), thickness=1)
            centers.append([centerX, centerY])
            #if cN == 0:
                ##=============Kalman Filter==================================
                #mp = np.array([[np.float32(centerX)],[np.float32(centerY)]])
                #kalman.correct(mp)
                #tp = kalman.predict()
                #cv.rectangle(mask, (int(tp[0]) - 5, int(tp[1]) - 5), (int(tp[0]) + 5, int(tp[1]) + 5), (0, 255, 255), 3);
                ##===========================================================
                #cN = cN + 1

    #s = KalmanManager.addPoints(centers)
    #print(s)
    #for x, y in s:
    #    cv.rectangle(mask, (int(x) - 5, int(y) - 5), (int(x) + 5, int(y) + 5), (255, 0, 255), 3)


    #closest = [0, 0]
    #for [cX, cY] in centers:
    #    l1 = getlength(closest[0], closest[1], cX, cY)
    #    l2 = getlength(closest[0], closest[1], previous[0], previous[1])
    #    
    #    if l2 != 0 or l1 < l2:
    #        closest = [centerX,centerY]
    #
    #previous = closest
    #closest = np.array(closest[0],closest[1])
    #kalman.correct(closest)
    #tp = kalman.predict()
    #cv.rectangle(mask, (int(tp[0]) - 5, int(tp[1]) - 5), (int(tp[0]) + 5, int(tp[1]) + 5), (0, 255, 255), 1);
    #pred.append((int(tp[0]),int(tp[1])))

    mx1 = 0
    my1 = 0
    mx2 = 0 
    my2 = 0

    for [x1, y1] in centers:
        for [x2, y2] in centers:
            if (x1 != x2 and y1 != y2):
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if (length < 120 and length > 80):
                    cv.line(mask, (math.floor(x1), math.floor(y1)), (math.floor(x2), math.floor(y2)), (255,0,0), thickness=1)
                    mx1 = x1
                    my1 = y1
                    mx2 = x2
                    my2 = y2

    
    cv.circle(mask, (math.floor(centerWidth), math.floor(centerHeight)), 10, (0,0,255), thickness=1)

    centerOfEarsX = (mx2 + mx1)/2
    centerOfEarsY = (my2 + my1)/2
    #gradient = (my2 - my1) / (mx2 - mx1)
    #gradientNormal = -(1/gradient)
    alpha = math.atan2(-(mx2 - mx1), (my2 - my1))

    newX = 600 * math.cos(alpha)
    newY = 600 * math.sin(alpha)

    centerofHeadX1 = centerOfEarsX + newX
    centerofHeadY1 = centerOfEarsY + newY

    centerofHeadX2 = centerOfEarsX - newX
    centerofHeadY2 = centerOfEarsY - newY

    length1 = math.sqrt((centerofHeadX1 - centerWidth)**2 + (centerofHeadY1 - centerHeight)**2)
    length2 = math.sqrt((centerofHeadX2 - centerWidth)**2 + (centerofHeadY2 - centerHeight)**2)
    
    centerOfHeadX = 0
    centerOfHeadY = 0
    if length1 > length2:
        centerOfHeadX = centerofHeadX1
        centerOfHeadY = centerofHeadY1
    else:
        centerOfHeadX = centerofHeadX2
        centerOfHeadY = centerofHeadY2

    #points = KalmanManager.addPoints([[centerOfEarsX,centerOfEarsY],[centerOfHeadX,centerOfHeadY]])

    #for x, y in points:
    #    cv.rectangle(mask, (int(x) - 5, int(y) - 5), (int(x) + 5, int(y) + 5), (255, 0, 255), 3)
    
    

    cv.line(frame, points[0], points[1], (255,0,0), thickness=1)
    #cv.line(frame, (math.floor(points[0][0]), math.floor(points[0][1])), (math.floor(points[1][0]), math.floor(points[1][1])), (255,0,0), thickness=1)
    seconds = frameNumber//fps
    cv.putText(frame,'Seconds: {}'.format(seconds),(10,500), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv.LINE_AA)
    
    cv.putText(frame,'GT Tracking: {}'.format("N/A"),(10,550), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)

    gtTracking = False
    for [l,h] in groundtruth:
        if seconds >= l and seconds <= h:
            gtTracking = True

    cv.putText(frame,'Predicted Tracking: {}'.format(gtTracking),(10,600), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)

    #centerOfEarsX = (mx2 + mx1)/2
    #centerOfEarsY = (my2 + my1)/2

    #y = mx + c
    #gradient = (my2 - my1) / (mx2 - mx1)
    #gradientNormal = -(1/gradient)

    #hx1 = centerOfEarsX + 20
    #hx2 = centerOfEarsX - 20
    #hy1 = (hx1 * gradient) + 
    


    #gradientX = (mx2 - mx1) / (my2 - my1)
    #gradientY = (my2 - my1) / (mx2 - mx1)

    #centerOfHeadX = centerOfEarsX + -(1/gradientX) * 10
    #centerOfHeadY = centerOfEarsY + -(1/gradientY) * 10

    #cv.line(mask, (math.floor(centerOfEarsX), math.floor(centerOfEarsY)), (math.floor(centerOfHeadX), math.floor(centerOfHeadY)), (255,0,0), thickness=1)

    #ret, markers = cv.connectedComponents(mask)
    
    #print(np.amax(markers))
    

    # computing K-Means with K = 2 (2 clusters)
    #centroids,_ = kmeans(data.astype(float),3)
    # assign each sample to a cluster
    #idx,_ = vq(data,centroids)
    #count = np.bincount(idx)
    #print(count)
    #kMeans = MiniBatchKMeans(n_clusters=5)
    #try:
    #    kMeans.fit(data)
    #    centers = kMeans.cluster_centers_
    #    
    #    for c in centers:
    #        cv.circle(mask, (math.floor(c[1]), math.floor(c[0])), 20, (0,0,255), thickness=1)
    #except ValueError:
    #    print("Frame Number {} : Too few datapoints for number of clusters".format(frameNo))

    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, mask)
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

cap.release()