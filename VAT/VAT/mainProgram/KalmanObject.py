import cv2 as cv
import numpy as np
import math

obj = []
number = 3
reInitCoords = False

def drawPredictions(frame):
    if reInitCoords == False:
        for kalman in obj:
            kalman.drawPrevPrediction(frame)

def getlength(coord1, coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)

def initCoords(coords):
    global obj
    obj = []
    for count, coord in enumerate(coords):
        obj.append(KalmanObj(coord, count))

def orderCoords(coords):
    global reInitCoords, number
    if len(coords) != number:
        reInitCoords = True
        return coords

    if reInitCoords:
        initCoords(coords)
        reInitCoords = False

    order = np.zeros((3,2), dtype = np.int)
    predictions = []

    for kalmanObj in obj:
        prediction = kalmanObj.prevPrediction
        predictions.append([prediction[0], prediction[1]])

    chosen = []

    for coord in coords:
        element = -1
        length = -1
        
        for count, prediction in enumerate(predictions):
            #print("{} {}".format(count, prediction))
            newLength = getlength(prediction, coord)
            if (element == -1 or length > newLength) and count not in chosen:
                length = newLength
                element = count

        chosen.append(element)
        obj[element].correct(convertCoord(coord))
        obj[element].predictNext()
        order[element] = coord
    
    return order

class KalmanObj:
    
    def __init__(self, initCoord, identification):
        self.initializeKalman(initCoord)
        print(identification)
        self.id = identification

    def initializeKalman(self, coord):
        self.kalman = cv.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        for i in range(20):
            self.kalman.correct(convertCoord(coord))
            self.predictNext()

    def predictNext(self):
        prediction = self.kalman.predict()
        self.prevPrediction = prediction
        return prediction

    def correct(self, coord):
        self.kalman.correct(convertCoord(coord))

    def drawPrevPrediction(self, frame):
        cv.circle(frame, (self.prevPrediction[0], self.prevPrediction[1]), 10, (0,0,255), thickness=1)
        cv.putText(frame,"{}".format(self.id) ,(self.prevPrediction[0], self.prevPrediction[1]), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1,cv.LINE_AA)

    def getPrevPrediction(self):
        return self.prevPrediction

def convertCoord(coord):
    return np.array([[np.float32(coord[0])],[np.float32(coord[1])]])
