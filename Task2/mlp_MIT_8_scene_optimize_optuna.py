import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import optuna
import wandb

#Put your key
wandb.login(key='4c0b25a1f87331e99edadbaa2cf9568a452224ff')


def model(params, number):
  #user defined variables
  IMG_SIZE    = int(params['image_size'])
  BATCH_SIZE  = int(params['batch_size'])
  DEPTH = int(params['depth'])
  REG = float(params['reg'])
  #ACTIVATION = LeakyReLU(alpha=0.01)
  ACTIVATION = 'relu'
  DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
  RESULTS = '/ghome/group06/lab2_C3_G06/results/last_layer_test/'
  MODEL_FNAME = RESULTS+params['size_layer2']+'_'+params['size_layer3']+'.weights.h5'

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
    shuffle=True,
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
    model.add(Dense(units=int(params["size_layer1"]), activation=ACTIVATION, name='first'))

    if DEPTH > 1:
      model.add(Dense(units=int(params["size_layer2"]), activation=ACTIVATION, name='second'))

      if DEPTH > 2:
        model.add(Dense(units=int(params["size_layer3"]), activation=ACTIVATION, name='third'))

  model.add(Dense(units=8, activation='softmax',name='classification'))
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  print(model.summary())
  plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

  if os.path.exists(MODEL_FNAME):
    print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

  print('Start training...\n')

  checkpoint = ModelCheckpoint(RESULTS+params['size_layer2']+'_'+params['size_layer3']+'_BEST.weights.h5', monitor='val_acc', verbose=1,
    save_best_only=True, save_weights_only=True, mode='auto')

  history = model.fit(
          train_dataset,
          epochs=100,
          validation_data=validation_dataset,
          verbose=0)


  print('Saving the model into '+MODEL_FNAME+' \n')
  model.save_weights(MODEL_FNAME)  # always save your weights after training or during training

    # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(RESULTS+str(number)+'accuracy.jpg')
  plt.close()
    # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(RESULTS+str(number)+'loss.jpg')
  plt.close()
  #to get the output of a given layer
  #crop the model up to a certain layer
  #layer = 'first'
  #model_layer = keras.Model(inputs=input, outputs=model.get_layer(layer).output)

  #Get the features from images
  #directory = DATASET_DIR+'/test/coast'
  #x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
  #x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
  #print(f'prediction for image {os.path.join(directory, os.listdir(directory)[0] )} on  layer {layer}')
  #features = model_layer.predict(x/255.0)
  #print(features.shape)
  #print(features)

  #Get classification
  #classification = model.predict(x/255.0)
  #print(f'classification for image {os.path.join(directory, os.listdir(directory)[0] )}:')
  #print(classification/np.sum(classification,axis=1))

  print('Done!')

  return history

def objective_model_cv(trial):

  
  params = {
    'image_size' : trial.suggest_categorical('image_size', ["32"]),
    "size_layer1": trial.suggest_categorical("size_layer1", ["256"]),
    "size_layer2": trial.suggest_categorical("size_layer2", ["128", "256"]),
    "size_layer3": trial.suggest_categorical("size_layer3", ["256", "512", "1024", "2048"]),
    'batch_size' : trial.suggest_categorical('batch_size', ["16"]),
    'depth' : trial.suggest_categorical('depth', ["3"]),
    'size' : trial.suggest_categorical('size', ["1"]),
    'reg' : trial.suggest_categorical('reg', ['0.01'])
  }
  

  config = dict(trial.params)
  config['trial.number'] = trial.number

  execution_name = 'Lab2_tests'+ str(trial.number)

  wandb.init(
      project = 'Lab2-tests-last-layer2',
      entity = 'c3_mcv',
      name = execution_name,
      config = config,
      reinit = True,
  )
  history = model(params, trial.number)

  #trial.report(accuracy, trial.number)
  # report validation accuracy to wandb
  for epoch in range(len(history.history['accuracy'])):
    wandb.log({
      'Epoch': epoch,
      'Train Loss': history.history['loss'][epoch],
      'Validation Loss': history.history['val_loss'][epoch],
      'Train Accuracy': history.history['accuracy'][epoch],
      'Validation Accuracy': history.history['val_accuracy'][epoch],
    })

  wandb.log(data={"Mean Accuracy": history.history['val_accuracy'][-1]})
  #return accuracy

study = optuna.create_study(direction="maximize", study_name = 'Lab_2')
study.optimize(objective_model_cv, n_trials=100)