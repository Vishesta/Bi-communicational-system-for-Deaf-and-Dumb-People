import cv2
import numpy as np
import math
import sys
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def generate_name(length = 3):
    char_list = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split()
    name = []
    for i in range(length):
        name.append(char_list[np.random.randint(len(char_list))])
    return "".join(name)

def getNewLabel(name,value = 'Dataset'):
	current_directory = os.getcwd()
	dataset_directory = os.path.join(current_directory,value)
	if not os.path.exists(dataset_directory):
		os.makedirs(dataset_directory)
	final_directory = os.path.join(dataset_directory, name)
	subdirectory_1 = os.path.join(final_directory, 'Original')
	if not os.path.exists(final_directory):
		os.makedirs(final_directory)
		os.makedirs(subdirectory_1)  	
	return subdirectory_1

capt = 0

while(True):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    start = 200
    end = 450

    cv2.rectangle(img, (start,start), (end,end), (102,185,255),5)
    crop_img = img[start:end, start:end]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    _, thresh1 = cv2.threshold(blurred, 135, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernal = np.ones((10,10),np.uint8)
    erosion = cv2.erode(thresh1,kernal,iterations=1)
    dilation = cv2.dilate(erosion,kernal,iterations=1)

    image, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    dilation_rgb = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)

    all_img = np.hstack((drawing, crop_img,dilation_rgb))
    cv2.imshow('WebCam', img)
    cv2.imshow('All', all_img)

    k = cv2.waitKey(10)

    if k == 27:
        break

    elif k == ord('c') or k == ord('C'):
        capt=1
        break

if(capt==1):
    cap.release()
    cv2.destroyAllWindows()
    print('Enter label name for this capture: ',end="")
    newlabel = input()
    original_dir = getNewLabel(newlabel)
    number = len(os.listdir(original_dir)) + 1

    original_img_path = os.path.join(original_dir, generate_name()+'.jpg')

    cv2.imwrite(original_img_path,crop_img)
    #cv2.imwrite(threshold_img_path,dilation)
    #cv2.imwrite(contour_img_path,drawing)