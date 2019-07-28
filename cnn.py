import cv2
import os
import sys
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def image_resize(image, height = 45, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if(w>h):
    	r = height / float(h)
    	dim = (int(w * r), height)
    else:
    	r = height / float(w)
    	dim = (height, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def data_read(data_path,category='Original',source='Dataset'):
	labels = []
	images = []
	data_path = os.path.join(data_path,source)
	for label in os.listdir(data_path):
		label_path = os.path.join(data_path,label,category)
		for image in os.listdir(label_path):
			img = cv2.imread(os.path.join(label_path,image))
			if(img is not None):
				resized_img = image_resize(img)
				images.append(resized_img)
				labels.append(label)
	return labels, images

labels, images = data_read(r'C:\Users\user\ISL')
data = pd.DataFrame({'Image Data':images,'Label':labels})
print('Data Sample')
print(data.head(10))


label_cats = np.unique(labels)
int_encoding = np.arange(len(label_cats))
label_to_int = pd.DataFrame({'Label':label_cats,'Encoded':int_encoding})
label_to_int.to_csv('label_encoded.csv')

label_convert = dict(zip(label_cats,int_encoding))

integer_encoded = np.array([label_convert[l] for l in labels])

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
labels_encoded = onehot_encoder.fit_transform(integer_encoded)

np.random.seed(2)

model = Sequential()
print(np.shape(images[0]))
model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=np.shape(images[0])))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(len(labels_encoded[0]),activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 200

history = model.fit(np.array(images), labels_encoded, epochs=epochs, batch_size=32)

model.save('trained.h5')