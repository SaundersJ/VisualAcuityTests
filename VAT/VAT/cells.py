import cv2 as cv
import numpy as np
from scipy import ndimage

window_capture_original = 'Original'
window_capture_after = 'After'
window_trackbar = "trackbar"

cv.namedWindow(window_capture_original)
cv.namedWindow(window_capture_after)
cv.namedWindow(window_trackbar)

img = cv.imread("C:/Users/Jack/Desktop/python/Project/contrast adjusted.tif")
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow(window_capture_original, img)

ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2, 5)

#ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(),255,0)
ret, sure_fg = cv.threshold(dist_transform, 0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0


def nothing(x):
    pass
cv.createTrackbar("threshVal", window_trackbar , 0, 20, nothing)
cv.setTrackbarPos("threshVal", window_trackbar, 0)
cv.createTrackbar("kSize", window_trackbar , 0, 30, nothing)
cv.setTrackbarPos("kSize", window_trackbar, 30)

cv.createTrackbar('Number of Superpixels', window_trackbar, 400, 10000, nothing)
cv.setTrackbarPos("Number of Superpixels", window_trackbar, 10000)
cv.createTrackbar('Iterations', window_trackbar, 4, 12, nothing)
cv.setTrackbarPos("Iterations", window_trackbar, 12)

#
#while True:
#    i = cv.getTrackbarPos("thresh", window_trackbar)
#    ret,th1 = cv.threshold(img, i ,255,cv.THRESH_BINARY)
#    cv.imshow(window_capture_after, th1)
#    key = cv.waitKey(30)
#
#    if key == ord('q') or key == 27:
#        break

#ret,th1 = cv.threshold(img, 31 ,255,cv.THRESH_BINARY)


##kSize
##Thresh

while True:
    threshVal = cv.getTrackbarPos("threshVal", window_trackbar)
    kSize = (cv.getTrackbarPos("kSize", window_trackbar) * 2) + 1
    #print(kSize)
    print(threshVal)
    
    #blur = cv.GaussianBlur(img,(kSize, kSize),0)


    #th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,21,0)

    #th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    #th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,21,0)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,kSize,threshVal)
    
    #ret,th1 = cv.threshold(img, threshVal ,255,cv.THRESH_BINARY)
    

    ###====================SEEDS

    image_width = 1392
    image_height = 1040
    image_channels = 1
    num_superpixels = cv.getTrackbarPos('Number of Superpixels', window_trackbar)
    num_levels = 4
    num_iterations = cv.getTrackbarPos('Iterations', window_trackbar)

    seeds = cv.ximgproc.createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels)
    seeds.iterate(th3, num_iterations)

    labels = seeds.getLabels()
    lab = labels.copy()
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)

    mask = seeds.getLabelContourMask(True)

    mask_inv = cv.bitwise_not(mask)
    result_bg = cv.bitwise_and(th3, th3, mask=mask_inv)

    color_img = np.zeros((image_height,image_width,3), np.uint8)
    color_img[:] = (255, 0, 0)


    result_fg = cv.bitwise_and(color_img, color_img, mask=mask)
    result_bg = cv.cvtColor(result_bg,cv.COLOR_GRAY2BGR)
    result = cv.add(result_bg, result_fg)

    spNoColours = {}
    spSize = {}


    num = 0

    finalImage = np.zeros(th3.shape)

    for l in range(lab.max()):
        labCopy = lab.copy()
        th3Copy = th3.copy()
        labCopy[labCopy != l]=0
        size = sum(sum(labCopy))
        final = labCopy * (th3Copy // 255)
        colour = sum(sum(final))
        #print("Percent of {} is {}".format(l, (colour/size)))
        if (colour/size) > 0.93:
            num = num + 1
            finalImage = finalImage + labCopy
            
    cv.imshow(window_capture_after, finalImage)
    print(num)    

    #for x in range(image_height - 1):
    #    for y in range(image_width - 1):
    #        colour = th3[x,y]
    #        spNo = l[x, y]
    #            
    #        if colour == 255:
    #            foundC = False
    #            for key, value in spNoColours.items():
    #                if key == spNo:
    #                    spNoColours[key] = spNoColours[key] + 1
    #                    foundC = True
    #                    break
    # 
    #             if not foundC:
    #                 spNoColours[spNo] = 1

    #        foundS = False
    #        for key, value in spSize.items():
    #            if key == spNo:
    #                spSize[key] = spSize[key] + 1
    #                foundS = True
    #                break
    # 
    #        if not foundS:
    #            spSize[spNo] = 1

    #print(spNoColours)
    #print(spSize)

    
    
    #numberOfCells = 0

    #for x in range(l.max()):
    #    numberOfColours = spNoColours[x]
    #    size = spSize[x]
    #    if numberOfColours / size > 0.5:
    #        numberOfCells = numberOfCells + 1
    #    
    #print(numberOfCells)
    ###======================

    



    #frameInv = cv.bitwise_not(th3)
    #mask = cv.dilate(frameInv, (5,5), iterations=2)
    #mask = cv.bitwise_not(frameInv)
    #mask = cv.erode(frameInv, (5,5), iterations=2)

    #ret, markers = cv.connectedComponents(frameInv)
    
    #print(markers.max())
    #markers = markers+1


    #mask = cv.erode(frameInv, None, iterations=2)
    #mask = cv.dilate(frameInv, None, iterations=2)
    #mask = cv.bitwise_not(mask)
    #ret,th1 = cv.threshold(blur,threshVal,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #ret,th1 = cv.threshold(blur, threshVal ,255,cv.THRESH_BINARY)

    
    # Add the masked foreground and background.



    ##Clustering
    #clusters = hcluster.fclusterdata(data, thresh, criterion="distance")


    #cv.imshow(window_capture_after, result)


    key = cv.waitKey(0)

    if key == ord('q') or key == 27:
        break



#blur = cv.GaussianBlur(img,(3,3),0)
#ret3,th3 = cv.threshold(blur,32,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#th3 = cv.adaptiveThreshold(th1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,7)



#cv.imshow(window_capture_after, th3)
#cv.waitKey(0)
#ret,th1 = cv.threshold(img,0,255,cv.THRESH_BINARY)


