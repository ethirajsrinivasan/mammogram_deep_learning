
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("mammo1.png")
blurred = cv2.pyrMeanShiftFiltering(image,31,91)
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
# In[2]:


# denos = cv2.fastNlMeansDenoising(image,None,10,7,21)


# In[3]:

ret,thresh = cv2.threshold(gray,10,255,0)


# In[4]:


_, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

contourAreas = [cv2.contourArea(contour) for contour in contours ]
    
largest_contour = contourAreas.index(max(contourAreas))
cv2.drawContours(image,contours,largest_contour,(0,0,255),6)


# In[5]:


cv2.namedWindow('Display',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow("Display",image)
cv2.waitKey(0)
