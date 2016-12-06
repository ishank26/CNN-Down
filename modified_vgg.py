# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:16:00 2016

@author: ishank
"""

# %load modified_vgg.py
#from datetime import datetime
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, random
import numpy as np
#import sys
from keras.callbacks import History
#import keras.callbacks as cb

np.random.seed(1707)



def loadData(datafile):
    with open(datafile) as f:
        files = f.readlines()
        img = []
        label = []
        for lines in files:
            item = lines.split()
            img.append(item[0])
            label.append(int(item[1]))
        return img, label

def loadImages(path_list, crop_size=224, shift=15):
    images = np.ndarray([len(path_list),3, crop_size, crop_size])
    for i in xrange(len(path_list)):
        img = cv2.imread(path_list[i])
        h, w, c = img.shape
        assert c==3
        img = img.astype(np.float32) 
        # cv2 load images as bgr, subtract mean accordingly
        #img[:,:,0] -= 103.939
        #img[:,:,1] -= 116.779
        #img[:,:,2] -= 123.68
        # transpose according to rgb
        #img_crop = img[shift:shift+crop_size, shift:shift+crop_size,:]
        img= img.transpose((2,0,1))
        images[i] = img
    return images

def savehist(file,loss,acc):
    with open(file,"a") as f:
        f.write(loss,"\n")
        f.write(acc,"\n")
        
#print loadImages(train_path)

'''
im = cv2.resize(cv2.imread('/home/neo/Desktop/work/cnn_down/data/align_224/box/go_norvgg_align_89.jpg'), (224, 224)).astype(np.float32)
print im.shape
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
print im.shape
im = im.transpose((2,0,1))
print im.shape
im = np.expand_dims(im, axis=0)
print im.shape
'''


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu',trainable= False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',trainable= False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',trainable= False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',trainable= False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable= False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable= False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable= False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def pop_layers(model,n):
    for i in range(n):
        model.layers.pop()
    return model



def shuffleData(data, labels):
    pos= [i for i in range(len(labels))]
    np.random.shuffle(pos)
    data=[data[i] for i in pos]
    labels=[labels[i] for i in pos]
    return data, labels


#set model params
nb_classes = 2
epochs_per_iter= 1
batch_size = 50


#load training data
print("Loading train data")
train_path, train_labels = loadData("/home/neo/Desktop/work/cnn_down/vgg_keras/train2.txt")


#shuffle
train_path, train_labels=shuffleData(train_path, train_labels)

#conv to matrix
X_train=loadImages(train_path)
y_train = np_utils.to_categorical(train_labels, nb_classes)


#load test data
print("Loading test data")
test_path, test_labels = loadData("/home/neo/Desktop/work/cnn_down/vgg_keras/test2.txt")

#shuffle
test_path,test_labels = shuffleData(test_path, test_labels)

#conv to matrix
X_test=loadImages(test_path)
y_test = np_utils.to_categorical(test_labels, nb_classes)



# train model

print "Adding model weights"
model = VGG_16('vgg16_weights.h5')
print "Model weights added"
#print model.layers


print "Poping specified layers"
model=pop_layers(model, 5)
#print model.layers

print "Rebuilding model"
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


#Set learning params
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

print "Compiling Model"
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
#store loss history
history = History()

#fit model
print "Fitting Model"
model.fit(X_train, y_train, shuffle=True, nb_epoch=epochs_per_iter, batch_size=batch_size, callbacks = [history])



print('Saving weights')
# save weights to load later
#model.save_weights('weights_vgg_mod.hdf5', overwrite=True)

#evaluate model
print "Testing"
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#plot loss profile  
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("loss per epoch")
plt.show()
plt.plot(history.history['acc'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title("accuracy per epoch")
plt.show()

#save history
vgg_loss=history.history['loss']
vgg_acc=history.history['acc']
savehist("hist.txt",vgg_loss,vgg_acc)
