# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.figure(0)
#
# # Original Image..........................
# img=cv2.imread("test_subject-2.jpeg")
# plt.subplot(1,3,1)
# plt.imshow(img)
# plt.title('Original Image')
#
# # color filter............................
# dim_grey = np.array([0,0,0])
# light_grey= np.array([240,240,240])
# mask = cv2.inRange(img,dim_grey,light_grey)
# clr_fltr=cv2.bitwise_and(img,img,mask=mask)
# plt.subplot(1,3,2)
# plt.imshow(clr_fltr)
# plt.title('color filter')
#
# # Erosion.....................................
# kernel=np.ones((3,3),np.uint8)
# ersn=cv2.erode(clr_fltr,kernel,iterations=1)  # iteration can be taken 2 if watershed is unavailable
# plt.subplot(1,3,3)
# plt.imshow(ersn)
# plt.title('Erosion')
#
# plt.figure(1)
#
# # Dialation.................................
# dltn=cv2.dilate(ersn,kernel,iterations=1)
# plt.subplot(1,3,1)
# plt.imshow(dltn)
# plt.title('Dilation')
#
#
# # Watershed..................................
# # wtrsd=cv2.watershed(dltn,markers=??????????) # could not find it
# # plt.subplot(1,3,2)
# # plt.imshow(wtrsd)
# # plt.title('Watershed')
#
# # croping.....................................
# gray = cv2.cvtColor(dltn, cv2.COLOR_BGR2GRAY)
# gray = cv2.blur(gray, (11,11))
# thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# # contours,hierarchy = cv2.findContours(gray, 1, 2)
# for cnt in contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     if w>100 and h>50:
#         break
# # plt.imshow(contours)
# crop = dltn[y:y+h, x:x+w]
# plt.subplot(1,3,3)
# plt.imshow(crop)
# plt.title('crop')
#
# plt.figure(2)
# # Median..........................................
# median = cv2.medianBlur(crop, 7)
# plt.subplot(1,3,1)
# plt.imshow(median)
# plt.title('Median')
#
# # Threshold Binary................................
# median = cv2.cvtColor(median,cv2.COLOR_BGR2GRAY)
# ret1,threshold1 = cv2.threshold(median,140,255,cv2.THRESH_BINARY)
# # ret1,threshold1 = cv2.threshold(im1,60,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
# ret2,threshold2 = cv2.threshold(median,20,255,cv2.THRESH_BINARY)
# plt.subplot(1,3,2)
# plt.imshow(threshold1)
# plt.title('Hemorrhage area')
# plt.subplot(1,3,3)
# plt.imshow(threshold2)
# plt.title('Brain Area')
#
# # area count..................................................
# area1 = cv2.countNonZero(threshold1)
# area2 = cv2.countNonZero(threshold2)
# print("Hemorrhage area = ",area1)
# print('Total brain area = ',area2,'\nPercentage of Hemorrhage = ',area1*100/area2,'%')
# plt.show()

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0)
img=cv2.imread("hmrg_brain.jpg")
#img=cv2.imread('normal_brain.jpg')
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original Image')

# color filter................................
dim_grey = np.array([0,0,0])
light_grey= np.array([250,250,250])
excluding_part = cv2.inRange(img,dim_grey,light_grey)
clr_fltr=cv2.bitwise_and(img,img,mask=excluding_part)
plt.subplot(2,2,2)
plt.imshow(clr_fltr)
plt.title('color filter')

# Erosion.....................................
kernel=np.ones((3,3),np.uint8)
ersn=cv2.erode(clr_fltr,kernel,iterations=5)
plt.subplot(2,2,3)
plt.imshow(ersn)
plt.title('Erosion')

# Dialation...................................
dltn=cv2.dilate(ersn,kernel,iterations=1)
plt.subplot(2,2,4)
plt.imshow(dltn)
plt.title('Dilation')

plt.figure(1)
# Watershed...................................

# croping.....................................
gray = cv2.cvtColor(dltn, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (11,11))
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
contours,hierarchy = cv2.findContours(thresh, 1, 2)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w>300 and h>300:
        break
crop = dltn[y:y+h, x:x+w]
plt.subplot(2,2,1)
plt.imshow(crop)
plt.title('crop')

# Median.....................................
median = cv2.medianBlur(crop, ksize=1)
plt.subplot(2,2,2)
plt.imshow(median)
plt.title('Median')

# Threshold Binary..........................
median = cv2.cvtColor(median,cv2.COLOR_BGR2GRAY)
ret1,threshold1 = cv2.threshold(median,150,255,cv2.THRESH_BINARY)
ret3,threshold1 = cv2.threshold(threshold1,50,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
ret4,threshold1 = cv2.threshold(threshold1,50,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
ret2,threshold2 = cv2.threshold(median,0,255,cv2.THRESH_BINARY)
plt.subplot(2,2,3)
plt.imshow(threshold1)
plt.title('Hemorrhage area')
plt.subplot(2,2,4)
plt.imshow(threshold2)
plt.title('Brain Area')

# area count.................................
area1 = cv2.countNonZero(threshold1)
area2 = cv2.countNonZero(threshold2)
print("Hemorrhage area = ",area1)
print('Total brain area = ',area2)
print('Percentage of Hemorrhage = ',area1*100/area2,'%')
plt.show()
