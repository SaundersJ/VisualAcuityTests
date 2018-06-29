import cv2 as cv

def thresholdEars(mask):
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
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:3]   #3 biggest areas
