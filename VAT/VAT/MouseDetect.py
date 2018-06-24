import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math
from scipy.cluster.vq import kmeans,vq
import LogUtil
import zipapp

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
#IMG_0845_GT
fileName = "C:/Users/Jack/Desktop/python/Project/IMG_0849.MOV"
LogUtil.init(fileName)
cap = cv.VideoCapture(fileName)

trackingTimes = []

previous = [100000, 100000]
angles = []

drumAverage = []
drumMinMaxAverage = []

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

def getlength(coord1, coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)

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

    im2, cnts, hierarchy = cv.findContours(mask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:3]

    centers = []

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv.boundingRect(approx)
        rect = (x, y, w, h)
        cv.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 1);
        centerX = math.floor(x + (w/2))
        centerY = math.floor(y + (h/2))
        cv.circle(mask, (centerX, centerY), 10, (0,0,255), thickness=1)
        centers.append([centerX, centerY])

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


    

    
    cv.line(frame, (math.floor(centerOfEarsX), math.floor(centerOfEarsY)), (math.floor(centerOfHeadX), math.floor(centerOfHeadY)), (255,0,0), thickness=1)
    cv.line(frame, (math.floor(centerOfEarsX), math.floor(centerOfEarsY)), (math.floor(previous[0]), math.floor(previous[1])), (255,0,255), thickness=1)
    
    ##Calculate Drum Rotation

    squareWidth = 70    

    point1 = (50, int(centerHeight) - int((squareWidth / 2)))
    point2 = (50, int(centerHeight) + int((squareWidth / 2)))

    cv.rectangle(frame, point1, point2, (0, 255, 255), 5);
    #start:end
    startSplice = int(centerHeight) - int((squareWidth / 2))
    endSplice = int(centerHeight) + int((squareWidth / 2))

    gray = cv.cvtColor(frame ,cv.COLOR_BGR2GRAY)
    ret, thg = cv.threshold(gray, 0, 255 ,cv.THRESH_BINARY+cv.THRESH_OTSU)

    drumAverage.append(int((np.mean(thg[startSplice:endSplice][0]) / 255) * 100))

    if (len(drumAverage) > 10):
        drumAverage.pop(0)    


    change = []

    for i in range(0, len(drumAverage) - 1):
        change.append(abs(drumAverage[i] - drumAverage[(i + 1)]))
    
    if len(change) == 0:
        change.append(0)
    
    
    minMaxChange = max(change) - min(change)

    drumMinMaxAverage.append(minMaxChange)

    if len(drumMinMaxAverage) > 20:
        drumMinMaxAverage.pop(0)

    #print("{:.} {:.2f} {}".format(meanChange, avgChange, drumAverage))

    #print(np.std(drumAverage))

    #meanChange = max(change) - min(change)
    #print(meanChange)
    #meanChange = np.mean(change)
    
    #changeStd = np.std(change)
    #rstd = changeStd / meanChange
    #print("{}".format(rstd))
    
    #print(np.std(drumAverage))
    #print(drumAverage)
    drumSpinning = True 
    if (len(drumAverage) != 10 or max(drumMinMaxAverage) < 3):
        drumSpinning = False
        #print("NotSpinning {}".format(drumAverage))
        cv.putText(frame,'Not Spinning'.format(alpha),(0,30), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)
        #print("Not Spinning {}".format(max(drumMinMaxAverage)))
    else:
        cv.putText(frame,'Spinning'.format(alpha),(0,30), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)
    ##

    ##    


    ##calculate angle - cosine rule -> http://www.cimt.org.uk/projects/mepres/step-up/sect4/index.htm

    coord1 = [centerOfHeadX, centerOfHeadY]
    coord2 = previous
    coord3 = [centerOfEarsX, centerOfEarsY]

    a = getlength(coord1, coord2)
    b = getlength(coord2, coord3)
    c = getlength(coord1, coord3)

    crossProduct = np.cross(coord1, coord2) #driection
    if crossProduct != 0:
        direction = crossProduct / abs(crossProduct)
    else:
        direction = 0

    alpha = math.acos((b**2 + c**2 - a**2)/(2 * b * c))
    alpha = direction * alpha

    alphaRevPerFrame = alpha / (2 * math.pi)
    alphaRevPerSec = alphaRevPerFrame * fps
    angles.append(alphaRevPerSec)
    if (len(angles) > 15):
        angles.pop(0)
    
    avg = np.mean(angles)
    standardDeviation = np.std(angles)
    
    absAvg = abs(avg)
    
    relativeSD = (standardDeviation / absAvg) * 100 #percent
    

    ##

    drumSpeed = 2 / 60 #rev/s
    percent = 0.30
    
    following = False
    if (drumSpinning and (absAvg < drumSpeed + (drumSpeed * percent)) and (absAvg > drumSpeed - (drumSpeed * percent)) and relativeSD < 300):
        following = True
        print("{} {} Tracking rSTD:{}".format(frameNumber/fps, frameNumber, relativeSD))
        LogUtil.write("{0:.2f}s Tracking rSTD: {1}".format((frameNumber/fps), relativeSD))
        trackingTimes.append(frameNumber/fps)

    
    
    #print("Compare: Drum {} and mouse {}  and {}".format(drumSpeed, avg, following))    

    #cv.putText(frame,'Rev/s: {}'.format(alpha),(-10,550), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)

    previous = [centerOfHeadX, centerOfHeadY]

    seconds = frameNumber//fps
    cv.putText(frame,'Seconds: {}'.format(seconds),(10,500), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv.LINE_AA)
    
    cv.putText(frame,'Predicted Tracking: {}'.format(following),(10,550), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)

    gtTracking = False
    for [l,h] in groundtruth:
        if seconds >= l and seconds <= h:
            gtTracking = True

    cv.putText(frame,'GT Tracking: {}'.format(gtTracking),(10,600), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)

    cv.imshow(window_capture_name, frame)
    #cv.imshow(window_capture_name, thg)
    cv.imshow(window_detection_name, mask)
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
            

    #changeTime = [j-i for i, j in zip(trackingTimes[:-1], trackingTimes[1:])]
    #print("track: {}".format(trackingTimes))
    #print("forma: {}".format(changeTime))

    #if len(changeTime) > 1:
    #    groupStart = 0
    #    groupCount = 1

#        threshold = 0.5
#        dic = {}#
#
#        #[11, 11.003, 11.226, 11.738, 11.758]
#        #[0.003, 0.223, 0.512, 0.02]#
#
#        for count, i in enumerate(changeTime):
#            if i > threshold:
#                #Create a new group
#                print("{}-{} count: {}".format(trackingTimes[groupStart], trackingTimes[groupStart + groupCount - 1], groupCount))
#                groupStart = count
#                groupCount = 1
#            else:
#                groupCount = groupCount + 1
    
    #for count, i in enumerate(changeTime):
    #    if i > threshold:
    #        #New group
    #        #currentCount = currentCount + 1
    #        print("{}-{} count: {}".format(trackingTimes[current], trackingTimes[current + currentCount], currentCount))
    #        #groups.append(["{}-{} count:".format(trackingTimes[current - 1], [trackingTimes[current - 1 + currentCount - 1], currentCount])
    #        
    #        current = count - 1
    #        currentCount = 1
    #    else:
    #        currentCount = currentCount + 1

    #print(dic)
            
cap.release()

groupStart = 0
groupCount = 1

prev = 0
for count, i in enumerate(trackingTimes):
    if groupStart == 0:
        groupStart = i
    else:
        if i - prev > 0.5:
            #above threshold
            LogUtil.writeToResults("{} - {} count: {}".format(groupStart, prev, groupCount))
            #print("{} - {} count: {}".format(groupStart, prev, groupCount))
            groupStart = i
            groupCount = 1
        else:
            groupCount = groupCount + 1
    prev = i

####Do stuff with the times.......
####Be able to add more videos....