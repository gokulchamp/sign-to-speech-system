#import tensorflow as tf

import os
import glob
import numpy as np
import pandas as pd
import cv2
import math
from skimage import draw
from PIL import ImageOps
import tensorflow as tf


model=tf.keras.models.load_model("models\mobilenetv2_model")
sign_dict={'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'A': 9, 'B': 10, 'C': 11, 'D': 12, 'E': 13, 'F': 14, 'G': 15, 'H': 16, 'I': 17, 'J': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'O': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30, 'W': 31, 'X': 32, 'Y': 33, 'Z': 34}

sign_dict={v:k for k,v in sign_dict.items()}




model=tf.keras.models.load_model("saved_models/sign_recognizer")
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
cap=cv2.VideoCapture(0)


def stackImages(scale, imgArray):    #stack images together
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]*2
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContours(imgHsv):
    contours, hierarchy = cv2.findContours(imgHsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 4)
    #print(contours)
    result=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = 5000 #cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dist=math.dist((x, y),(w,h))
            result=img[y:y+h,x:x+w]
    return result


def hsv_converter(img):

    blur = cv2.GaussianBlur(img, (3,3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82]) 
    upper_color = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    hsv_d = cv2.dilate(blur, kernel)

    
    return hsv_d,hsv



#removal of face
def face_detector(imgFace):
    faceCascade = cv2.CascadeClassifier('DATA/haarcascade_classifier_frontal_face.xml')
    gray = cv2.cvtColor(imgFace, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE) ##flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    mask = np.zeros(shape=img.shape[0:2], dtype="bool")

    
    for (x, y, w, h) in faces:
        height,width,c=img.shape
        

        dist=math.dist((x,y), (x+w, y+h))
        rr, cc = draw.rectangle(start=(x,y), end=(x+w, y+h+int(dist/3))) # Draw a filled rectangle on the mask image
        mask[cc, rr] = True
        imgcopy[mask]=0  #applying the black mask on the image to remove the face
       
    return imgcopy
    



while(True):

    success,img=cap.read()
    #img=cv2.resize(img,(img.shape[1],img.shape[1]))
    img=cv2.flip(img,1)
    imgcopy=img.copy()

    img_Faceremoved=face_detector(imgcopy)
    imgHsv,hsv=hsv_converter(img_Faceremoved)

    




    hand_image=getContours(imgHsv)
    hand_image=cv2.resize(hand_image,(224,224))
    imgStack = stackImages(0.75, ([img,img_Faceremoved],[hsv,imgHsv]))
    if not np.any(hand_image):
        print("nothing")
        cv2.putText(img,"Nothing",(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    else:
        image=np.array([hand_image])/255.
        predictions = model.predict(image)
        result=predictions.tolist()[0]
        position=result.index(max(result))
        sign=sign_dict[position] 
        print(sign)
        
        cv2.putText(img,sign,(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        

    cv2.imshow("hand_image",hand_image)
    cv2.imshow("image", imgStack)
    cv2.imshow("Result",result)
    cv2.moveWindow("image",300,100)
    
    cv2.resizeWindow("image",(1000,800))

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
