import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math
from scipy.cluster.vq import kmeans,vq
import LogUtil
import zipapp
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import os.path
import KalmanObject

##Methods
def getlength(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def getlength(coord1, coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)

acceptedFileExtensions = [".MOV", ".mp4"]

def getMouseDistinction(mask, viewVideo):
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

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    centers = []

    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv.boundingRect(approx)
        rect = (x, y, w, h)
        if viewVideo:
            cv.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 1);
        centerX = math.floor(x + (w/2))
        centerY = math.floor(y + (h/2))
        if viewVideo:
            cv.circle(mask, (centerX, centerY), 10, (0,0,255), thickness=1)
        centers.append([centerX, centerY])

    return centers, mask

def getTrackingTimes(file, viewVideo, gui):
    ##Inits
    if not os.path.exists(file):
        gui.logMessage("{} is not a File, please enter a correct file location containing videos".format(file))
        return None
    
    fileNames = os.listdir(file)
    videos = []
    for f in fileNames:
        filename, file_extension = os.path.splitext(f)
        if file_extension in acceptedFileExtensions:
            videos.append(file + '\\' + f)

    if len(videos) < 1:
        gui.logMessage("{} does not contain any valid files, please use from the following extensions: {}".format(file, acceptedFileExtensions))
        return None


    if viewVideo:
        window_capture_name = 'Video Capture'
        window_detection_name = 'Object Detection'

        cv.namedWindow(window_capture_name)
        cv.namedWindow(window_detection_name)
        

    gui.logMessage("Starting Tracking for files {}".format(videos))
    track = []

    for count, fileName in enumerate(videos):
        #gui.totalStatusBar.setValue(int((count / len(videos)) * 100))
        #gui.currentStatusBar.setValue(0)
        fileViewStartTime = datetime.now()
        gui.logMessage("\n\n")
        gui.logMessage("Init for file: {}".format(fileName))
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
        
        centers, mask = getMouseDistinction(frame, viewVideo)
        
        KalmanObject.initCoords(centers)

        height, width = frame.shape[:2]
        centerWidth = width / 2
        centerHeight = height / 2
        fps = math.ceil(cap.get(cv.cv2.CAP_PROP_FPS))
    
        print(datetime.now())

        while True:
            frameNumber = frameNumber + 1
            ret, frame = cap.read()
            
            if frame is None:
                print("Break {}".format(frameNumber))
                break

            mask = frame
            centers, mask = getMouseDistinction(mask, viewVideo)            
            
            centers = KalmanObject.orderCoords(centers)
            KalmanObject.drawPredictions(mask)            

            mx1 = 0
            my1 = 0
            mx2 = 0
            my2 = 0

            ear1 = 0
            ear2 = 0
                    
            positionCoords = [0, 1, 2]
            tail = -1
            
            

            for count1, [x1, y1] in enumerate(centers):
                if tail != -1:
                    break

                for count2, [x2, y2] in enumerate(centers):
                    if (x1 != x2 and y1 != y2):
                        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if (length < 120 and length > 80):
                            if viewVideo:
                                cv.line(mask, (math.floor(x1), math.floor(y1)), (math.floor(x2), math.floor(y2)), (255,0,0), thickness=1)

                            mx1 = x1
                            my1 = y1
                            mx2 = x2
                            my2 = y2
                            ear1 = count1
                            ear2 = count2
                            positionCoords.remove(ear1)
                            positionCoords.remove(ear2)
                            tail = positionCoords[0]
                            break



            if viewVideo:
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

            #length1 = math.sqrt((centerofHeadX1 - centerWidth)**2 + (centerofHeadY1 - centerHeight)**2)
            #length2 = math.sqrt((centerofHeadX2 - centerWidth)**2 + (centerofHeadY2 - centerHeight)**2)

            length1 = 0
            length2 = 0

           
            if len(centers) != 3:
                continue

            length1 = math.sqrt((centerofHeadX1 - centers[tail][0])**2 + (centerofHeadY1 - centers[tail][1])**2)
            length2 = math.sqrt((centerofHeadX2 - centers[tail][0])**2 + (centerofHeadY2 - centers[tail][1])**2)

            centerOfHeadX = 0
            centerOfHeadY = 0

            #tailLocation = centers[tail]
            #cv.circle(frame, (math.floor(centerofHeadX1), math.floor(centerofHeadY1)), 10, (0,0,255), thickness=1)
            #cv.circle(frame, (math.floor(centerofHeadX2), math.floor(centerofHeadY2)), 10, (0,0,255), thickness=1)

            if length1 > length2:
                centerOfHeadX = centerofHeadX1
                centerOfHeadY = centerofHeadY1
            else:
                centerOfHeadX = centerofHeadX2
                centerOfHeadY = centerofHeadY2

            #Calculate which line is futhest from the tail

            

            if viewVideo:
                cv.line(mask, (math.floor(centerOfEarsX), math.floor(centerOfEarsY)), (math.floor(centerOfHeadX), math.floor(centerOfHeadY)), (255,0,0), thickness=1)
                cv.line(mask, (math.floor(centerOfEarsX), math.floor(centerOfEarsY)), (math.floor(previous[0]), math.floor(previous[1])), (255,0,255), thickness=1)
    
            ##Calculate Drum Rotation

            squareWidth = 70    

            point1 = (50, int(centerHeight) - int((squareWidth / 2)))
            point2 = (50, int(centerHeight) + int((squareWidth / 2)))
            if viewVideo:
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

            drumSpinning = True 
            if (len(drumAverage) != 10 or max(drumMinMaxAverage) < 3):
                drumSpinning = False
                if viewVideo:
                    cv.putText(frame,'Not Spinning'.format(alpha),(0,30), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)
            else:
                if viewVideo:
                    cv.putText(frame,'Spinning'.format(alpha),(0,30), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)

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
        
            try:
                alpha = math.acos((b**2 + c**2 - a**2)/(2 * b * c))
                alpha = direction * alpha
            except ValueError:
                value = (b**2 + c**2 - a**2)/(2 * b * c)
                if value > 0:
                    alpha = 0
                else:
                    alpha = 180

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
  

            previous = [centerOfHeadX, centerOfHeadY]

            seconds = frameNumber//fps
            if viewVideo:
                cv.putText(frame,'Seconds: {}'.format(seconds),(10,500), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv.LINE_AA)
        
            if viewVideo:
                cv.putText(frame,'Predicted Tracking: {}'.format(following),(10,550), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv.LINE_AA)

            key = 1
            if viewVideo:
                cv.imshow(window_capture_name, frame)
                cv.imshow(window_detection_name, mask)
                key = cv.waitKey(30)

            if key == ord('q') or key == 27:
                break
            if frameNumber % 100 == 0:
                #gui.currentStatusBar.setValue(int((frameNumber / int(cap.get(cv.CAP_PROP_FRAME_COUNT))) * 100))
                print(frameNumber / int(cap.get(cv.CAP_PROP_FRAME_COUNT)))

        cap.release()
        print(datetime.now())
        print("That viewing took {}".format(datetime.now() - fileViewStartTime))







        groupStart = 0
        groupCount = 1
        prev = 0
        pos = 0
    
        totalTime = 0
        totalGroups = 0
        for count, i in enumerate(trackingTimes):
            if groupStart == 0:
                groupStart = i
            else:
                if i - prev > 0.5:
                    #above threshold
                    diff = prev - groupStart
                    groupTotal = diff * fps
                    groupTotal = int(round(groupTotal, 0) + 1)
                    LogUtil.writeToResults(pos, 0, "{:.2f} - {:.2f} prob: {} / {}".format(groupStart, prev, groupCount, groupTotal))
                    pos = pos + 1
                    #print("{} - {} count: {}".format(groupStart, prev, groupCount))
                    groupStart = i
                    groupCount = 1

                    totalTime = totalTime + diff
                    totalGroups = totalGroups + 1
                else:
                    groupCount = groupCount + 1
            prev = i
    
        pos = pos + 2 
        LogUtil.writeToResults(pos, 0, "Total Time: {:.2f}".format(totalTime))
        pos = pos + 1
        LogUtil.writeToResults(pos, 0, "Total Stares: {}".format(totalGroups))

        track.append(trackingTimes)
    return track, videos