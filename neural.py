# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:34:46 2022

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

x=[]
l0 = []
l1 = []
new_paths = []

paths = df.iloc[:,-1].values
lane0 = df.iloc[:,df.columns.str.startswith('lanes/0')].values
lane1 = df.iloc[:,df.columns.str.startswith('lanes/1')].values
lane2 = df.iloc[:,df.columns.str.startswith('lanes/2')].values
lane3 = df.iloc[:,df.columns.str.startswith('lanes/3')].values
lane4 = df.iloc[:,df.columns.str.startswith('lanes/4')].values
h_sample = df.iloc[:,df.columns.str.startswith('h_sample')].values


print(len(lane0[0]))

full0 = np.asarray(np.where((lane0 < 0).sum(axis=1) <= 0))
full1 = np.asarray(np.where((lane1 < 0).sum(axis=1) <= 0))

match = (np.intersect1d(full0, full1))

#print(len(full0[0]))
    
for i in full0:
    #print(i)
    l0 = lane0[i,:]
    new_paths = paths[i]
    
for i in full1:
    #print(i)
    l1 = lane1[i,:]
    #new_paths = paths[i]    

    
lane = np.concatenate((lane0, lane1, lane2, lane3), axis=1)
#lane = np.reshape(lane, (4, len(lane0), len(lane0[1])))


for j in range(0, len(paths)):

    
    p = paths[j]
    p = 'train_set/' + p
    #p = 'train_set/clips/0531/1492626287507231547/20.jpg'
    
    image = cv2.imread(p)
    img = image[160:710:10,0:1000,:]
    
    #cv2.imshow('image',img)
    #cv2.waitKey(1000)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img/255
    
    #cv2.imshow('image', img)
    #cv2.waitKey(1000)
    
    #img = np.resize(img, (90,160,3))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x.append(img)
    
#cv2.destroyAllWindows()

x = np.array(x)
y = lane0

x_vals=[]
y_vals=[]

coef_matrix=[]
#print(len(y))

for i in range(0, len(y)):
    for j in range(0, len(y[1])):
        #print(i,j)
        if(y[i,j]!=-2 and np.isnan(y[i,j]) == False):
            x_vals.append(y[i,j])
            y_vals.append(h_sample[i,j])
    X = np.array(x_vals)
    Y = np.array(y_vals)
    x_vals.clear()
    y_vals.clear()
    coef = np.polyfit(Y,X,1)
    coef_matrix.append(coef)      

coef_matrix = np.array(coef_matrix)
y = coef_matrix
num_output = len(y[0])

# for i in range(0,len(coef_matrix)):
#     for j in range(0,len(coef_matrix[1])):
#         if(coef_matrix[i][j] > -.1 and coef_matrix[i][j] < 0):
#             coef_matrix[i][j] = 0
#         if(coef_matrix[i][j] > )

def lane_net():
    
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(66, 200, 3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(15, (3, 3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(10, (3, 3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(5, (3, 3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(150))
    model.add(LeakyReLU(alpha=0.05))
    
    model.add(Dense(125))
    model.add(LeakyReLU(alpha=0.05))
    
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.05))
    
    #model.add(Dropout(0.5))
    model.add(Dense(num_output))
    model.add(LeakyReLU(alpha=0.05))
    # Compile model
    model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
    #model.compile(Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
#model = lane_net()
#print(model.summary())

#history = model.fit(x,y,validation_split=0.3,epochs=5,batch_size=50, verbose=1)


def nvidia_model():
      model = Sequential()
      model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(66,200,3),activation='tanh'))
 
      model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='tanh'))
      model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='tanh'))
      model.add(Conv2D(64, kernel_size=(3,3), activation='tanh'))
      model.add(Conv2D(64, kernel_size=(3,3), activation='tanh'))
      #model.add(Dropout(0.5))
 
 
      model.add(Flatten())
      model.add(Dense(100, activation='tanh'))
      model.add(Dropout(0.5))
 
 
      model.add(Dense(50, activation='tanh'))
      model.add(Dropout(0.5))
      
      model.add(Dense(25, activation ='tanh'))
      model.add(Dropout(0.5))
      
      model.add(Dense(num_output))
 
      optimizer= Adam(learning_rate=1e-5)
      model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
 
      return model
  
model = nvidia_model()
print(model.summary())

history = model.fit(x,coef_matrix,validation_split=0.3, epochs=50,batch_size=5, verbose=1, shuffle=True)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# model.save('allpaths1.h5')
    
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

#import random

# x_pred = []
# index = random.randint(0,len(paths))
# rand_path = paths[index]
# rand_path = 'train_set/' + rand_path
# #p = 'train_set/clips/0531/1492626287507231547/20.jpg'

# image = cv2.imread(rand_path)

# img = image[160:710:10,0:1000,:]

# #cv2.imshow('image',img)
# #cv2.waitKey(1000)

# img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
# img = cv2.GaussianBlur(img, (3,3), 0)
# img = cv2.resize(img, (200,66))
# img = img/255

# x_pred.append(img)
# x_pred = np.array(x_pred)
# y = h_sample[0]

# model = keras.models.load_model('newpaths_93.h5')


# pred = model.predict(x_pred)
# pred = np.resize(pred, (56))

# #solver = np.poly1d(coef)
# #x_val = solver(y)

# for i in range(0,len(y)):
#     #print(pred[i])
#     cv2.drawMarker(image, (int(pred[i]),int(y[i])), (0,0,255), markerSize = 10, thickness =2)
#     #print(i)

# cv2.imshow('image',image)
# cv2.waitKey(0)
       
# cv2.destroyAllWindows()

# print(coef_matrix[0:4,:])
# 

x=[]

for j in range(0, len(paths)):
    p = paths[j]
    p = 'train_set/' + p
    #p = 'train_set/clips/0531/1492626287507231547/20.jpg'
    
    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x.append(image)
    
x = np.array(x)

def unet():
    inputs = Input((1280,720,1))
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
    model.compile(optimizer=Adam(lr=1e-3), loss='mse', metrics =['accuracy'])
    return model

model = unet()
print(model.summary())
