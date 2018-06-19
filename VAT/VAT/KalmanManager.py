import cv2 as cv
import numpy as np
import math

previousCoords = []
previousPredict = []
filters = []
number = 0
##arguments, list of coords
##[[1,1],[1,1],[1,1]]
def initKalman(coords):
    global previousCoords, previousPredict, filters, number
    number = len(coords)
    for x, y in coords:
        previousCoords.append([x, y])
        kalman = cv.KalmanFilter(4,2)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        filters.append(kalman)

        mp = np.array([[np.float32(x)],[np.float32(y)]])
        kalman.correct(mp)
        tp = kalman.predict()
        previousPredict.append([tp[0], tp[1]])

def getlength(coord1, coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)

def addPoints(coords):
    if len(coords) == number:
        global previousCoords, previousPredict, filters
    
        ##Loop through previous coords to get the closest coordinates to it
        i = 0
        ##position of previously defined features
        listOfChosenCoords = []
        for pX, pY in previousCoords:
            ##coordinates closest to the previous coordinates
            sX = 100000
            sY = 100000
            chosenCoordPos = 0
            posJ = 0
            for x, y in coords:
                if (getlength([pX, pY], [x, y]) < getlength([pX, pY], [sX, sY]) and (chosenCoordPos not in listOfChosenCoords)):
                    sX = x
                    sY = y
                    chosenCoordPos = posJ
                posJ = posJ + 1
        
            print(chosenCoordPos)
            previousCoords[i] = [sX, sY]
            kalman = filters[i]
            mp = np.array([[np.float32(sX)],[np.float32(sY)]])
            kalman.correct(mp)
            tp = kalman.predict()
            previousPredict[i] = ([tp[0], tp[1]])

            coords[chosenCoordPos] = [100000, 100000]
            i = i + 1

            listOfChosenCoords.append(posJ)
    
        #print(previousPredict)
    else:
        for i, kalman in enumerate(filters):
            tp = kalman.predict()
            previousPredict[i] = ([tp[0], tp[1]])
        print("NOT ENOUGH DATAPOINTS")
    return previousPredict
        
        


    #for i in range(n):
    #    previousCoords.append([10000, 10000])
    #    kalman = cv.KalmanFilter(4,2)
    #    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    #    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    #    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    #    filters.append(kalman)

    #previousCoords = previousCoords * 10000 
    #number = n
    #print(previousCoords)



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
    