import os
import requests
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Dropout
from keras.utils import plot_model
from keras import regularizers

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import optuna
import wandb
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from sklearn.svm import SVC


#Put your key
#wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')

#Function to fit and get the accuracies on SVM
def SVM_layer(model, IMG_SIZE, classifier):
  directory_train = DATASET_DIR+'/train'
  directory_test = DATASET_DIR+'/test'
  classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
  
  feats_train = []
  feats_test = []
  labels_train = []
  labels_test = []

  for class_dir in os.listdir(directory_train):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory_train,class_dir)):
      im = np.asarray(Image.open(os.path.join(directory_train,class_dir,imname)))
      im = np.expand_dims(np.resize(im, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
      out = model.predict(im/255., verbose=0)
      feats_train.append(np.array(out).flatten())
      labels_train.append(cls)
  
  for class_dir in os.listdir(directory_test):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory_test,class_dir)):
      im = np.asarray(Image.open(os.path.join(directory_test,class_dir,imname)))
      im = np.expand_dims(np.resize(im, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
      out = model.predict(im/255., verbose=0)
      feats_test.append(np.array(out).flatten())
      labels_test.append(cls)

  classifier.fit(feats_train, labels_train)
  accuracy = classifier.score(feats_test, labels_test)
  return accuracy

params = {
  'image_size' : '32',
  'batch_size' : '16',
  'depth' : '3',
  'size_layer1' : '256',
  'size_layer2' : '128',
  'size_layer3' : '1024',
  'kernel': 'rbf'
  #'kernel': 'linear'
}


DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
RESULTS = '/ghome/group06/lab2_C3_G06/results/last_layer_test4/'
MODEL_FNAME = RESULTS+'128_1024_BEST.weights.h5'
#user defined variables
IMG_SIZE    = int(params['image_size'])
BATCH_SIZE  = int(params['batch_size'])
DEPTH = int(params['depth'])
#ACTIVATION = LeakyReLU(alpha=0.01)
SIZE_LAYER1 = int(params['size_layer1'])
SIZE_LAYER2 = int(params['size_layer2'])
SIZE_LAYER3 = int(params['size_layer3'])
ACTIVATION = 'relu'


if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()

if not os.path.exists(RESULTS):
  os.makedirs(RESULTS)


print('Setting up data ...\n')


# Load and preprocess the training dataset
train_dataset = keras.utils.image_dataset_from_directory(
  directory=DATASET_DIR+'/train/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(IMG_SIZE, IMG_SIZE),
  shuffle=True,
  validation_split=None,
  subset=None
)

# Load and preprocess the validation dataset
validation_dataset = keras.utils.image_dataset_from_directory(
  directory=DATASET_DIR+'/test/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(IMG_SIZE, IMG_SIZE),
  shuffle=False,
  seed=123,
  validation_split=None,
  subset=None
)

# Data augmentation and preprocessing
preprocessing_train = keras.Sequential([
  keras.layers.Rescaling(1./255),
  keras.layers.RandomFlip("horizontal")
])

preprocessing_validation = keras.Sequential([
  keras.layers.Rescaling(1./255)
])

train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


print('Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
input = Input(shape=(IMG_SIZE, IMG_SIZE, 3,),name='input')
model.add(input) # Input tensor
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),name='reshape'))
if DEPTH > 0:
  #model.add(Dense(units=int(2048*float(params['size'])), kernel_regularizer=regularizers.l2(0.001), activation=ACTIVATION, name='first'))
  model.add(Dense(units=SIZE_LAYER1, activation=ACTIVATION, name='first'))

  if DEPTH > 1:
    #model.add(Dense(units=int(1024*float(params['size'])), kernel_regularizer=regularizers.l2(0.001), activation=ACTIVATION, name='second'))
    #model.add(Dropout(DO_FACTOR))
    model.add(Dense(units=SIZE_LAYER2, activation=ACTIVATION, name='second'))

    if DEPTH > 2:
      #model.add(Dense(units=int(512*float(params['size'])), kernel_regularizer=regularizers.l2(0.001), activation=ACTIVATION, name='third'))
      model.add(Dense(units=SIZE_LAYER3, activation=ACTIVATION, name='third'))
      
      
model.add(Dense(units=8, activation='softmax',name='classification'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

print('Loading weights from '+MODEL_FNAME+' ...\n')
print ('\n')

model.load_weights(MODEL_FNAME)

print('Done!\n')

#Get the model outputs for each hidden layer:
#First, get layer names:
layer_names = [layer.name for layer in model.layers]

#Now, get models from only the hidden layers (all except first and last layer):
layers= []
for layer_name in layer_names[1:-1]:
  model_layer = keras.Model(inputs=input, outputs=model.get_layer(layer_name).output)
  layers.append(model_layer)

#Declare SVM:
classifier = SVC(kernel=params['kernel'], random_state=42)
for model in layers:
  accuracy = SVM_layer(model, IMG_SIZE, classifier)
  print(f'The accuracy with kernel {str(params["kernel"])} is: {str(accuracy)}')

