# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:29:57 2022

@author: Naman Jain
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, UpSampling2D, BatchNormalization, Concatenate, Input
from keras.layers import *
from tensorflow.keras.optimizers import *


df = pd.read_csv('train_set/label_data_0531.csv')
h = 32
w = 64
ratio = 720/h

x=[]
l0 = []
l1 = []
new_paths = []
x_v = []
y_v = []
X=[]
Y=[]
coef_matrix=[]
X_mask=[]
orig=[]

order = 2

paths = df.iloc[:,-1].values
lane0 = df.iloc[:,df.columns.str.startswith('lanes/0')].values
lane1 = df.iloc[:,df.columns.str.startswith('lanes/1')].values
lane2 = df.iloc[:,df.columns.str.startswith('lanes/2')].values
lane3 = df.iloc[:,df.columns.str.startswith('lanes/3')].values
lane4 = df.iloc[:,df.columns.str.startswith('lanes/4')].values
h_sample = df.iloc[:,df.columns.str.startswith('h_sample')].values

h_sample = h_sample/ratio


#lane = np.concatenate((lane0), axis=1)
lane = np.resize(lane0, (358,1,56))

# LOADING IMAGES IN A NUMPY ARRAY
for j in range(0, len(paths)):
    p = paths[j]
    p = 'train_set/' + p
    #p = 'train_set/clips/0531/1492626287507231547/20.jpg'
    
    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig.append(image)
    image = np.resize(image, (h,w))
    x.append(image)  
x = np.array(x)
orig = np.array(orig)

#GENERATING POLYFIT FOR LABELS
for i in range(len(lane)):
    for j in range(len(lane[0])):
        for k in range(56):
            if(lane[i,j,k]!=-2 and np.isnan(lane[i,j,k]) == False):
                x_v.append(int(lane[i,j,k]/ratio))
                y_v.append(h_sample[i,k])
        X = np.array(x_v)
        Y = np.array(y_v)
        x_v.clear()
        y_v.clear()
        
        if(len(X) == 0):
            coef = np.zeros((order+1))
        else:
            coef = np.polyfit(Y,X,order)
        coef_matrix.append(coef) 
        solver = np.poly1d(coef)
        y_range = h_sample[0]
        x_pred = solver(y_range)
        X_mask.append(x_pred)        

coef_matrix = np.array(coef_matrix)
coef_matrix = np.resize(coef_matrix,(len(lane), len(lane[1]),order+1))
X_mask = np.array(X_mask)
X_mask = np.reshape(X_mask, (358,1,56))
#X_mask = np.reshape(X_mask, (358,4,551))
X_mask = X_mask.astype(int)


y_segment = np.zeros(x.shape)

#print(len(X_mask[1]))

#PUTTING 255 WHERE NEEDED ON MASK
for i in range(len(X_mask)):
    for j in range(len(X_mask[1])):
        for k in range(71-16+1):
            row_val = int(y_range[k])
            a = X_mask[i,j,k]
            #print(i,j,k,a,row_val)
            if(a>0 and a<w):
                y_segment[i,row_val,a] = 1
                #print('True')
                #print(y_segment[i,row_val,a])


for k in range(y_segment.shape[0]):
    for i in range(y_segment.shape[1]):
        flag = 0
        for j in range(y_segment.shape[2]):
            if(flag == 0 and y_segment[k][i][j]==1):
                flag = 1
            elif(flag==1 and y_segment[k][i][j]==0):
                y_segment[k][i][j] = 1
            elif(flag==1 and y_segment[k][i][j]==1):
                flag=0

import random
for i in range(50):
    print(i)
    rand = random.randint(0, len(X_mask)-2)
    im = orig[i,:,:]
    img = y_segment[rand,:,:]
    res = cv2.resize(img, (1280,720))
    #dst = cv2.addWeighted(res,1,im,1,0)
    cv2.imshow('image',res)
    cv2.imshow('image2',im)
    cv2.waitKey(0)
cv2.destroyAllWindows()

def unet():
    inputs = Input((h,w,1))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer="rmsprop", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics =['accuracy'])
    #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics =['accuracy'])
    return model

Model.add(Dense(50))

import sklearn

model = unet()
print(model.summary())

#x = np.resize(x, (358,32,64))
history = model.fit(x,y_segment,validation_split=0.2, epochs=5, batch_size=5, verbose=1, shuffle=True)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('unet5_all1_binarycross.h5')
    
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model = keras.models.load_model('unet5_all1_binarycross.h5')

import random
index = random.randint(0,len(paths))
print(index)
test_img = x[index,:,:].reshape(1,h,w,1)
pred = model.predict(test_img)
img = np.resize(pred, (h,w))
resized = cv2.resize(img, (1280,720))
orr = orig[index,:,:]
cv2.imshow('image',resized)
cv2.imshow('original',orr)
cv2.waitKey(0)         
cv2.destroyAllWindows()


np.save('test3.npy', a)
