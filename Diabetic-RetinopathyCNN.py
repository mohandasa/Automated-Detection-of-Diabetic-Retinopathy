# -*- coding: utf-8 -*-
#imports
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.datasets import fetch_olivetti_faces

from keras import backend as K

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import theano


from PIL import Image

from numpy import *
from theano import tensor as T
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#%%

olivetti = fetch_olivetti_faces()

#getting the inpuut data and preprocessed data
Input_data = "/Users/radhikashroff/Desktop/data/train"
preprocessed_data = "/Users/radhikashroff/Desktop/data/data_resized"

Imagelist =  os.listdir(Input_data)
Image_samples = size(Imagelist)

print Image_samples

for file in Imagelist:
    read_Image = Image.open(Input_data + "//" + file)
    resized_Image = read_Image.resize((img_rows, img_cols))
    gray = resized_Image.convert("L")
    
    gray.save(preprocessed_data + "//" + file, "JPEG")
    
preprocessedlist = os.listdir(preprocessed_data)   #preprocessed image data is read

image1 = array(Image.open("/Users/radhikashroff/Desktop/data/data_resized" + "/" + preprocessedlist[0]))
m,n = image1.shape[0:2]
imagepreprocessedlist = len(preprocessedlist)
    
print imagepreprocessedlist

imagematrix = array([array(Image.open("/Users/radhikashroff/Desktop/data/data_resized"+ '/' +image2)).flatten()
                    for image2 in preprocessedlist], 'f')  #fattening the images

label = np.ones((Image_samples,), dtype= int)
label[0:154]=0
label[154:308]=1
label[308:462]=2
label[462:616]=3
label[616:770]=4

data,Label= shuffle(imagematrix, label, random_state =2)
train_data = [data,Label]
resized_Image = imagematrix[20].reshape(img_rows, img_cols)  #reshaping the images
plt.imshow(resized_Image)
plt.imshow(resized_Image,cmap= "gray")
print(train_data[0].shape)
print(train_data[1].shape)

#%%

batch_size = 50
nb_classes = 5
nb_epoch = 2

img_rows, img_cols = 200,200

img_channels =1 #gray

nb_filters = 32

nb_pool =2 #pooling window size

nb_conv =3 #filter size

#%%

#splitting the data in train and test

(X,y) = (train_data[0], train_data[1])

train_X, test_X, train_Y, test_y = train_test_split(X,y, test_size = 0.2, random_state =2)

train_X = train_X.reshape(train_X.shape[0], 1, img_rows, img_cols)
test_X = test_X.reshape(test_X.shape[0], 1, img_rows, img_cols)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X /= 255
test_X /= 255

print("train_X shape:", train_X.shape)
print(train_X.shape[0], "train samples")
print(test_X.shape[0], "test samples")

train_Y = np_utils.to_categorical(train_Y,nb_classes)
test_Y = np_utils.to_categorical(test_y, nb_classes)

i =26
plt.imshow(train_X[i,0], interpolation = "nearest")
print ("label :", train_Y[i,:])

#%%

#Creating the model

model = Sequential()
model.add(Convolution2D(nb_filters,nb_conv, nb_conv, border_mode= "valid", input_shape = (1,img_rows, img_cols)))
convout1 = Activation("relu")
model.add(convout1)
model.add(Convolution2D(nb_filters,nb_conv, nb_conv))
convout2 = Activation("relu")
model.add(convout2)
model.add(MaxPooling2D(pool_size= (nb_pool, nb_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy",optimizer= "adadelta",metrics=["accuracy"])



#%%

#fitting the model 

model.fit(train_X, train_Y, batch_size= batch_size, nb_epoch= nb_epoch, show_accuracy = True, verbose= 1,
          validation_data= (test_X,test_Y))
 
model.fit(train_X, train_Y, batch_size= batch_size, nb_epoch= nb_epoch, show_accuracy = True, verbose= 1, 
          validation_split= 0.2)

#%%

#calculate the score and accuracy
score = model.evaluate(test_X, test_Y, show_accuracy =True, verbose= 0)

print('Test score:', score[0])
print('test_Accuracy:', score[1])
print(model.predict_classes(test_X[1:5]))
print(test_Y[1:5])


#%%

#getting the output of the hidden layer

get_feature = K.function([model.layers[0].input], [model.layers[0].output,])


input_image = train_X[0:1,:,:,:]
print(input_image.shape)

plt.imshow(input_image[0,0,:,:], cmap ='gray')
plt.imshow(input_image[0,0,:,:])

output_image = get_feature([input_image])

#X = np.array([output_image])

print(output_image[0].shape)

output_image = np.rollaxis(np.rollaxis(output_image[0],3,1),3,1)
print(output_image.shape)

fig = plt.figure(figsize=(8, 8))
#plt.imshow(output_image[0,:,:,i])

for i in range(32):
    ax = fig.add_subplot(6,6,i+1)
    ax.imshow(output_image[0,:,:,i])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    plt






 
