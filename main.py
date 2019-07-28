import cv2
import numpy as np
import pandas as pd
import math
import sys
import os
import tensorflow as tf 
from keras.models import load_model

from wordsegment import load,segment

cap = cv2.VideoCapture(0)
img_width = 1280
img_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

def image_resize(image, height = 45, inter = cv2.INTER_AREA):
    resized = cv2.resize(image, (height,height), interpolation = inter)
    return resized

model = load_model('trained.h5')

encoding_chart = pd.read_csv('label_encoded.csv')
encoding_values = encoding_chart['Encoded'].values
encoding_labels = encoding_chart['Label'].values
int_to_label = dict(zip(encoding_values,encoding_labels))

font = cv2.FONT_HERSHEY_DUPLEX

history = list()
counts = dict()
history_length = 15
threshold = 0.9

start = 200
end = 500
alpha = 0.4

sentence_raw = list()

color = (59, 185, 246)

load()

while(True):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    alpha_layer = img.copy()
    source = img.copy()

    crop_img = source[start:end, start:end]
    cv2.circle(alpha_layer, (int((start+end)/2),int((start+end)/2)), int((end - start)/2), color ,-1)
    cv2.addWeighted(alpha_layer, alpha, img, 1 - alpha,0, img)

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    resized = image_resize(crop_img)
    predicted = model.predict(np.array([resized]))

    predicted_char = int_to_label[np.argmax(predicted)]
    
    if(len(history)>=history_length):
        keys = list(counts.keys())
        values = list(counts.values())
        arg = np.argmax(values)
        if(values[arg]>threshold*history_length):
            sentence_raw.append(keys[arg])
        counts.clear()
        history.clear()
    if(predicted_char != 'None'):
        history.append(predicted_char)
        if(predicted_char in counts):
            counts[predicted_char]+=1
        else:
            counts[predicted_char]=1
        textsize = cv2.getTextSize(predicted_char, font, 6,7)[0]
        textX = int(start + ((end - start) - textsize[0])/2)
        textY = int(end - ((end - start) - textsize[1])/2)
        cv2.putText(img, predicted_char, (textX,textY),font,6,color,7)

    scribble = "".join(sentence_raw)
    sentence = " ".join(segment(scribble))
    sentencesize = cv2.getTextSize(sentence, font, 1,2)[0]

    if(len(sentence)>0):
        cv2.rectangle(img,(int((img_width - sentencesize[0])/2) - 20,img_height - 140),(int((img_width - sentencesize[0])/2 + sentencesize[0] + 20),img_height - 100 + sentencesize[1]),(0,0,0),-1)
    if(len(sentence)>30):
        sentence_raw = list(segment(scribble)[-1])

    cv2.putText(img, sentence, (int((img_width - sentencesize[0])/2),img_height - 100),font,1,(255,255,255),2)

    cv2.imshow('WebCam', img)
    k = cv2.waitKey(10)
    if k == ord('x') or k == ord('X'):
        sentence_raw.clear()
    if k == 27:
        break