import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
os.chdir('C:/Users/E0146968/Downloads/standardisedImages/')
images = glob.glob("*.jpg")
for im in images:
    print(im)
    image = cv2.imread(im)
    blurred = cv2.pyrMeanShiftFiltering(image,11,91)
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,10,255,0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contourAreas = [cv2.contourArea(contour) for contour in contours ]
    if len(contourAreas) > 0:
        largest_contour = contourAreas.index(max(contourAreas))
        mask = np.ones(image.shape[:2], dtype="uint8") * 255
        # loop over the contours
        for i,value in enumerate(contours):
            # if the contour is bad, draw it on the mask
            if i ==largest_contour:
                cv2.drawContours(mask, [value], -1, 0, -1)
        mask_inv = cv2.bitwise_not(mask)
        image = cv2.bitwise_and(image, image, mask=mask_inv)
        cv2.imwrite('../preprocess/'+im,image)
