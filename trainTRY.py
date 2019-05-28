import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import argparse
from keras.optimizers import Adam
import pickle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
import tensorflow as tf
class LeNet:
  @staticmethod
  def build(width, height, depth, classes):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)
 
    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)
    model.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(500))
    model.add(Activation("relu"))
 
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    model.summary()
    # return the constructed network architecture
    return model




data=[]
labels=[]
cnt=0
for path, subdirs, files in os.walk('./tempimages/'):
  for name in files:
    if cnt==0:
      cnt=cnt+1
      continue
    else:
      img_path = os.path.join(path,name)
      t1,t2=os.path.split(img_path)
      w,folder,correct_cat=t1.split('/')
      x,end=t2.split('.')
      if end=='jpg':
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        image = img_to_array(image)
        data.append(image)
        if correct_cat == "Accidents":
          label = 1
        elif correct_cat == "Fire":
          label = 2
        elif correct_cat=="NoAccidents":
          label = 0
        labels.append(label)
  cnt=cnt+1

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
  height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
  horizontal_flip=True, fill_mode="nearest")

EPOCHS = 25
INIT_LR = 1e-3
BS = 32


classifier=LeNet.build(width=64, height=64, depth=3, classes=3)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
classifier.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
classifier.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
  validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
  epochs=EPOCHS, verbose=1)
#pickle.dump(classifier,open('model.sav','wb'))

test_eval = classifier.evaluate(testX, testY, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

from keras.models import load_model 
classifier.save('trainTRY.h5')



