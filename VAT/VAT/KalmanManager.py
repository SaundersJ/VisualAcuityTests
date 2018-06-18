import cv2 as cv
import numpy as np


previous = []
filters = []
number = 0
def initKalman(n):
    for i in range(n):
        previous.append([1, 1])
        filters.append(cv.KalmanFilter(4,2))
    previous = previous * 10000 
    number = n

    for kalman in filters:
        kalman = cv.KalmanFilter(4,2)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03


def getlength(coord1, coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)



##https://stackoverflow.com/questions/3394835/args-and-kwargs
##arguments
##coordinate list....
#def addPoints(*args):
    
    #mp = np.array([[np.float32(centerX)],[np.float32(centerY)]])
    #kalman.correct(mp)
    #tp = kalman.predict()
    #cv.rectangle(mask, (int(tp[0]) - 5, int(tp[1]) - 5), (int(tp[0]) + 5, int(tp[1]) + 5), (0, 255, 255), 3);
        
    #dic = {}



    #for count, coord in enumerate(args):
    #    closest = -1
    #    for i, prev in enumerate(previous):
    #        if closest == -1:
    #            closest = count
    #        else:
    #            if getlength(prev, coord) <  

    #return newX1, newY1, newX2, newY2, newX3, newY3
    