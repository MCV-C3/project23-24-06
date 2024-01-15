import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
import optuna
import wandb
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC



#Put your key

params = {
    'patch_size' : '32',
    'batch_size' : "16",
    'depth' : "3",
    "size_layer1": "256",
    "size_layer2": "128",
    "size_layer3": "64",
    'size' : "1",
    'reg' :'0.01',
    'normalize': 0,
    'kernel' : ['rbf'],
    'vocab_size' : [300],
    'visualize': 'False',
  }

#user defined variables
IMG_SIZE = 256
PATCH_SIZE  = int(params['patch_size'])
BATCH_SIZE  = int(params['batch_size'])
DEPTH = int(params['depth'])
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
PATCHES_DIR = '/ghome/group06/lab2_C3_G06/C3/data/MIT_split_patches'+str(PATCH_SIZE)
MODEL_FNAME = '/ghome/group06/lab2_C3_G06/C3/patch_based_mlp'+str(PATCH_SIZE)+'_num_0.weights.h5'
CONF_MATRIX_PATH = '/ghome/group06/lab2_C3_G06/C3/Conf_Matrix/patch_based_mlp'+str(PATCH_SIZE)+'.weights.jpg'


if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()


Train_descriptors = []
Train_label_per_descriptor = []
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

def descriptors(model, img):

  patches = get_patches(img, PATCH_SIZE)
  descriptors = model.predict(patches/255., verbose=0)

  return descriptors

# Normalization
def l2_norm(histograms):
  histograms = histograms/np.linalg.norm(histograms)

  return histograms

# Scaler
def scaler(histograms):
  dev = np.std(histograms)
  histograms -= np.mean(histograms)
  histograms /= dev

  return histograms

#Clustering with KMeans:
def clustering(descriptors, codebook_size):
  codebook = MiniBatchKMeans(n_clusters=codebook_size, 
                             verbose=False, 
                             batch_size=codebook_size * 20,
                             compute_labels=False,
                             reassignment_ratio=10**-4,
                             random_state=42)
  codebook.fit(descriptors)

  return codebook

def generate_vw(vocab_size, normalize, descriptors, codebook):

  visual_words = np.zeros((len(descriptors), vocab_size), dtype=np.float32)

  for i in range(len(descriptors)):
    #Generate visual words
    words = codebook.predict(descriptors[i])

    hist = np.bincount(words, minlength=vocab_size)
    if normalize == 1:
      hist = l2_norm(hist)
    elif normalize == 2:
      hist = scaler(hist)

    #Add it
    visual_words[i,:] = hist

  return visual_words

def histogram_intersection_kernel(X, Y):
  # Compute the pairwise histogram intersection kernel matrix
  K = np.zeros((X.shape[0], Y.shape[0]))
  for i in range(X.shape[0]):
      for j in range(Y.shape[0]):
          K[i, j] = np.sum(np.minimum(X[i], Y[j]))

  return K

def classifier_sel(kernel):
  
  if kernel == 'histint':
    classif = SVC(kernel=histogram_intersection_kernel, random_state=42)
  else:
    classif = SVC(kernel=kernel, random_state=42)

  return classif

def BoVW(train_desc, train_labels, test_desc, test_labels, params):

  #Stack train descriptors
  D=np.vstack(train_desc)

  for vocab_size in params['vocab_size']:
    #Generate codebook
    codebook = clustering(D, vocab_size)

    #Generate visual words
    visual_words = generate_vw(vocab_size, params['normalize'], train_desc, codebook)
    visual_words_test = generate_vw(vocab_size, params['normalize'], test_desc, codebook)
    
    for kernel in params['kernel']:
      #Create and train knn
      classifier = classifier_sel(kernel)
      classifier.fit(visual_words, train_labels)

      #Predictions and scores
      predictions = classifier.predict(visual_words_test)
      accuracy = classifier.score(visual_words_test, test_labels)
      print(f'Accuracy for vocab_size {str(vocab_size)} and kernel {kernel}: {str(accuracy)}')
      #Visualize
      if params['visualize'] == 'True':
        classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

        #Confusion matrix
        confusion_matrix(classes, predictions, test_labels, CONF_MATRIX_PATH)


  return accuracy




print('Building MLP model for testing...\n')

model = Sequential()
input = Input(shape=(PATCH_SIZE, PATCH_SIZE, 3,),name='input')
model.add(input)
model.add(Reshape((PATCH_SIZE*PATCH_SIZE*3,)))
if DEPTH > 0:
  model.add(Dense(units=int(params["size_layer1"]), activation='relu', name='first'))

  if DEPTH > 1:
    model.add(Dense(units=int(params["size_layer2"]), activation='relu', name='second'))

    if DEPTH > 2:
      model.add(Dense(units=int(params["size_layer3"]), activation='relu', name='third'))

model.add(Dense(units=8, activation='softmax'))

print(model.summary())

print('Done!\n')

print('Loading weights from '+MODEL_FNAME+' ...\n')
print ('\n')

model.load_weights(MODEL_FNAME)

print('Done!\n')

print('Start feat extraction...\n')

#First, get layer names:
layer_names = [layer.name for layer in model.layers]
#print(layer_names[-2])
model_layer = keras.Model(inputs=input, outputs=model.get_layer(layer_names[-2]).output) #Get output of last hidden layer

directory_train = DATASET_DIR+'/train'
directory_test = DATASET_DIR+'/test'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}

feats_train = []
feats_test = []
labels_train = []
labels_test = []

#Get number of patches
num_patches = (IMG_SIZE // PATCH_SIZE)**2

for class_dir in os.listdir(directory_train):
  cls = classes[class_dir]
  for imname in os.listdir(os.path.join(directory_train,class_dir)):
    im = Image.open(os.path.join(directory_train,class_dir,imname))
    out = descriptors(model_layer, im)
    feats_train.append(out)
    labels_train.append(cls)

for class_dir in os.listdir(directory_test):
  cls = classes[class_dir]
  for imname in os.listdir(os.path.join(directory_test,class_dir)):
    im = Image.open(os.path.join(directory_test,class_dir,imname))
    out = descriptors(model_layer, im)
    feats_test.append(out)
    labels_test.append(cls)

print('Done!\n')

print('Start evaluation on BoVW...\n')

acc = BoVW(feats_train, labels_train, feats_test, labels_test, params)
