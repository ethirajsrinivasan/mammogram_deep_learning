
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import os
import glob


# In[4]:


os.chdir('C:/Users/E0146968/Downloads/dias_preprocess/')
images = glob.glob("*.png")
for img in images:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    ret,thresh = cv2.threshold(gray,1,255,0)
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image,contours,-1,(0,0,255),6)
    # cv2.namedWindow('Display',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 600,600)
    # cv2.imshow("Display",image)
    # cv2.waitKey(0)
    contourAreas = [cv2.contourArea(contour) for contour in contours ]
    largest_contour = contourAreas.index(max(contourAreas))
    cnt = contours[largest_contour]
    x,y,w,h = cv2.boundingRect(cnt)
    print(x,y,w,h)
    crop = image[y:y+h,x:x+w]
    res = cv2.resize(crop,(1024, 1024), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('../clear_border_ddsm/'+img,res)

