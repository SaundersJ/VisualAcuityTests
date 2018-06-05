import numpy as np
import cv2
import skimage as ski
import matplotlib.pyplot as plt
from sklearn.mixture import GMM

#https://github.com/llSourcell/Object_Detection_demo_LIVE/blob/master/Strawberry%20working.ipynb


def nothing(x):
    pass

cap = cv2.VideoCapture('C:/Users/Jack/Desktop/python/Project/IMG_0845.MOV')
cv2.namedWindow('image')
cv2.resizeWindow('image', 960, 540)

cv2.createTrackbar('r_lower','image',0,255,nothing)
cv2.createTrackbar('r_higher','image',0,255,nothing)
cv2.createTrackbar('g_lower','image',0,255,nothing)
cv2.createTrackbar('g_higher','image',0,255,nothing)
cv2.createTrackbar('b_lower','image',0,255,nothing)
cv2.createTrackbar('b_higher','image',0,255,nothing)



while(cap.isOpened()):
    ret, im = cap.read()
    cv2.resizeWindow('image', 960, 540)
    im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
    #frameGRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #th, frameThresh = cv2.threshold(frameGRAY, 90, 255, cv2.THRESH_BINARY)
    
    #h, w, _ = im.shape;       # Height and width of the image

    # Extract Blue, Green and Red
    #imB = im[:,:,0]; imG = im[:,:,1]; imR = im[:,:,2];

    # Reshape Blue, Green and Red channels into single-row vectors
    #imB_V = np.reshape(imB, [1, h * w]);
    #imG_V = np.reshape(imG, [1, h * w]);
    #imR_V = np.reshape(imR, [1, h * w]);

    # Combine the 3 single-row vectors into a 3-row matrix
    #im_V =  np.vstack((imR_V, imG_V, imB_V));

    # Calculate the bimodal GMM
    #nmodes = 2;
    #GMModel = GMM(n_components = nmodes, covariance_type = 'full', verbose = 0, tol = 1e-3)
    #model = GMM(n_components = nmodes, covariance_type = 'diag', verbose = 0, tol = 1e-3)
    #model = model.fit(np.transpose(im_V))

    #s = model.means_
    
    #print(s)

    #newdata = img_data.reshape(800*800, 4)
    #gmm = GaussianMixture(n_components=3, covariance_type="tied")
    #gmm = gmm.fit(frameGRAY)

    #cluster = gmm.predict(frameGRAY)
    #cluster = cluster.reshape(800, 800)
    #imshow(cluster)

    #frameGRAY = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    #thresh = cv2.adaptiveThreshold(frameGRAY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,41,7)
    
    #cv2.imshow('frame',thresh)
    #cv2.imshow('first', thresh)
    #cv2.imshow('second', thresh)

    #hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    #v = hsv[:,:,2]
    #s = hsv[:,:,1]
    #h = hsv[:,:,0]

    #lower = cv2.getTrackbarPos('hsvLower','image')
    #higher = cv2.getTrackbarPos('hsvHigher','image') 

    #th, frameThresh = cv2.threshold(v, lower, higher, cv2.THRESH_BINARY)

    #lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    #r = im[:,:,2]
    #g = im[:,:,1]
    #b = im[:,:,0]

    
    #th, frameThresh = cv2.threshold(im[:,:,x], cv2.getTrackbarPos('r_lower','image'), cv2.getTrackbarPos('hsvLower','image'), cv2.THRESH_BINARY)



    reR, threshR = cv2.threshold(im[:,:,0], cv2.getTrackbarPos('r_lower','image'), cv2.getTrackbarPos('r_higher','image'), cv2.THRESH_BINARY_INV)
    reG, threshG = cv2.threshold(im[:,:,1], cv2.getTrackbarPos('g_lower','image'), cv2.getTrackbarPos('g_higher','image'), cv2.THRESH_BINARY_INV)
    reB, threshB = cv2.threshold(im[:,:,2], cv2.getTrackbarPos('b_lower','image'), cv2.getTrackbarPos('b_higher','image'), cv2.THRESH_BINARY_INV)

    zipped = np.dstack((threshR, threshG, threshB))
    cv2.imshow('image', zipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print("Start")