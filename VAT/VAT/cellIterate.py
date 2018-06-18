import cv2 as cv

import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.color import rgb2grey
from skimage import io
import matplotlib.pyplot as plt

window_capture_original = 'Original'
window_capture_after = 'SEEDS'
window_trackbar = "trackbar"

cv.namedWindow(window_capture_original)
cv.namedWindow(window_capture_after)
cv.namedWindow(window_trackbar)

def nothing(*arg):
    pass

cv.createTrackbar('Number of Superpixels', window_trackbar, 400, 10000, nothing)
cv.createTrackbar('Iterations', window_trackbar, 4, 12, nothing)


img = cv.imread("C:/Users/Jack/Desktop/python/Project/contrast adjusted.tif")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

while True:
    image_width = 1392
    image_height = 1040
    image_channels = 1
    num_superpixels = cv.getTrackbarPos('Number of Superpixels', window_trackbar)
    num_levels = 4
    num_iterations = cv.getTrackbarPos('Iterations', window_trackbar)

    seeds = cv.ximgproc.createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels)
    seeds.iterate(img, num_iterations)

    labels = seeds.getLabels()
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)

    mask = seeds.getLabelContourMask(True)

    mask_inv = cv.bitwise_not(mask)
    result_bg = cv.bitwise_and(img, img, mask=mask_inv)

    color_img = np.zeros((image_height,image_width,1), np.uint8)
    color_img[:] = (255)


    result_fg = cv.bitwise_and(color_img, color_img, mask=mask)
    result = cv.add(result_bg, result_fg)






    #img2 = img.copy()
    #img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

    #cv.drawContours(img2, labels, 5, (0,255,0), 3)
    #cv.drawContours(img2, [labels.astype(int)], 0, (0, 255, 0), -1)

    cv.imshow(window_capture_after, result)
    cv.imshow(window_capture_original, img)

    key = cv.waitKey(30)

# Set the image into a grid, use the OTSU threshold for each gridded space
#numberOfGrids = 10



#ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)