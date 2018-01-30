import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
images = glob.glob("/media/ethi/Seagate Backup Plus Drive/AR/cases/*/*/*/*.jpg")
with open('ddsm_dataset.csv','w') as csvfile:
    writer = csv.writer(csvfile,delimiter=",")
    for image in images:
        im = cv2.imread(image)
        mean = cv2.mean(im)[0]
        blurred = cv2.pyrMeanShiftFiltering(im,11,91)
        gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,10,255,0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contourAreas = [cv2.contourArea(contour) for contour in contours ]
        max_contour = max(contourAreas)
        writer.writerow([image,mean,max_contour])