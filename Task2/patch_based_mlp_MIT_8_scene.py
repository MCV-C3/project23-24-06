import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
import optuna
import wandb

#Put your key
wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')

def get_patches(image, patch_size):
    # Get the size of the image
    width, height = image.size

    # Calculate the number of tiles in both dimensions
    num_tiles_x = width // patch_size
    num_tiles_y = height // patch_size

    # Crop the image into tiles
    patches = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            left = i * patch_size
            upper = j * patch_size
            right = (i + 1) * patch_size
            lower = (j + 1) * patch_size
            patch = np.array(image.crop((left, upper, right, lower)))
            patches.append(patch)

    return np.array(patches)

def descriptors(model, img, patch_size):

  patches = get_patches(img, patch_size)
  descriptors = model.predict(patches/255., verbose=0)

  return descriptors


def model(params, number):
  #user defined variables
  PATCH_SIZE  = int(params['patch_size'])
  BATCH_SIZE  = int(params['batch_size'])
  DEPTH = int(params['depth'])
  DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
  PATCHES_DIR = '/ghome/group06/lab2_C3_G06/C3/data/MIT_split_patches'+str(PATCH_SIZE)
  MODEL_FNAME = '/ghome/group06/lab2_C3_G06/C3/patch_based_mlp'+str(PATCH_SIZE)+'_num_'+str(number)+'.weights.h5'
  BEST_MODEL_FNAME = '/ghome/group06/lab2_C3_G06/C3/best_patch_based_mlp'+str(PATCH_SIZE)+'_num_'+str(number)+'.weights.h5'


  if not os.path.exists(DATASET_DIR):
    print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
    quit()
  if not os.path.exists(PATCHES_DIR):
    print('WARNING: patches dataset directory '+PATCHES_DIR+' does not exist!\n')
    print('Creating image patches dataset into '+PATCHES_DIR+'\n')
    generate_image_patches_db(DATASET_DIR,PATCHES_DIR,patch_size=PATCH_SIZE)
    print('patxes generated!\n')

  # Data augmentation and preprocessing
  preprocessing_train = keras.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.RandomFlip("horizontal")
  ])

  preprocessing_validation = keras.Sequential([
    keras.layers.Rescaling(1./255)
  ])

  # Load and preprocess the training dataset
  train_dataset = keras.utils.image_dataset_from_directory(
    directory=PATCHES_DIR+'/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    image_size=(PATCH_SIZE, PATCH_SIZE)
  )

  # Load and preprocess the validation dataset
  validation_dataset = keras.utils.image_dataset_from_directory(
    directory=PATCHES_DIR+'/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    image_size=(PATCH_SIZE, PATCH_SIZE)
  )

  train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
  validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

  train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



  def build_mlp(input_size=PATCH_SIZE,phase='train'):
    model = Sequential()
    model.add(Input(shape=(input_size, input_size, 3,),name='input'))
    model.add(Reshape((input_size*input_size*3,)))
    if DEPTH > 0:
      model.add(Dense(units=int(params["size_layer1"]), activation='relu', name='first'))

      if DEPTH > 1:
        model.add(Dense(units=int(params["size_layer2"]), activation='relu', name='second'))

        if DEPTH > 2:
          model.add(Dense(units=int(params["size_layer3"]), activation='relu', name='third'))
    if phase=='test':
      model.add(Dense(units=8, activation='linear')) # In test phase we softmax the average output over the image patches
    else:
      model.add(Dense(units=8, activation='softmax'))
    return model


  print('Building MLP model...\n')

  model = build_mlp(input_size=PATCH_SIZE)

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  print(model.summary())


  train = True
  if  not os.path.exists(MODEL_FNAME) or train:
    print('WARNING: model file '+MODEL_FNAME+' do not exists!\n')
    print('Start training...\n')
    
    # Define a ModelCheckpoint callback to save the best model weights
    checkpoint = ModelCheckpoint(BEST_MODEL_FNAME,
                                save_best_only=True,
                                save_weights_only=True,
                                monitor='val_accuracy',
                                mode='auto',
                                verbose=0)

    history = model.fit(train_dataset,
              epochs=150,
              validation_data=validation_dataset,
              callbacks=[checkpoint],
              verbose=0)
    
    print('Saving the model into '+MODEL_FNAME+' \n')
    model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
    print('Done!\n')


  print('Building MLP model for testing...\n')

  model = build_mlp(input_size=PATCH_SIZE, phase='test')
  print(model.summary())

  print('Done!\n')

  print('Loading weights from '+BEST_MODEL_FNAME+' ...\n')
  print ('\n')

  model.load_weights(BEST_MODEL_FNAME)

  print('Done!\n')

  print('Start evaluation ...\n')

  directory = DATASET_DIR+'/test'
  classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
  correct = 0.
  total   = 807
  count   = 0

  for class_dir in os.listdir(directory):
      cls = classes[class_dir]
      for imname in os.listdir(os.path.join(directory,class_dir)):
        im = Image.open(os.path.join(directory,class_dir,imname))
        #patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=1)
        #out = model.predict(patches/255., verbose = 0)
        out = descriptors(model, im, PATCH_SIZE)
        predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )
        if predicted_cls == cls:
          correct+=1
        count += 1
        #print('Evaluated images: '+str(count)+' / '+str(total), end='\r')
  
  test_acc = correct/total
  print('Done!\n')
  print('Test Acc. = '+str(correct/total)+'\n')

  return test_acc, history

def objective_model_cv(trial):

  
  params = {
    'patch_size' : trial.suggest_categorical('patch_size', ['32']),
    'batch_size' : trial.suggest_categorical('batch_size', ["16"]),
    'depth' : trial.suggest_categorical('depth', ["3"]),
    "size_layer1": trial.suggest_categorical("size_layer1", ["256"]),
    "size_layer2": trial.suggest_categorical("size_layer2", ["128"]),
    "size_layer3": trial.suggest_categorical("size_layer3", ["64"]),
    'size' : trial.suggest_categorical('size', ["1"]),
    'reg' : trial.suggest_categorical('reg', ['0.01'])
  }
  

  config = dict(trial.params)
  config['trial.number'] = trial.number

  execution_name = 'Lab2_tests'+ str(params['patch_size'])

  wandb.init(
      project = 'Lab2-patches-patch_size-final',
      entity = 'c3_mcv',
      name = execution_name,
      config = config,
      reinit = True,
  )
  test_acc, history = model(params, trial.number)

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

  wandb.log(data={"Test Accuracy": test_acc})
  #return accuracy

study = optuna.create_study(direction="maximize", study_name = 'Lab_2')
study.optimize(objective_model_cv, n_trials=1)
