#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Activation, AveragePooling2D, Input
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

# Function to compute the mean values of images in the dataset
def compute_dataset_mean(dataset_path):
  mean_r, mean_g, mean_b = 0.0, 0.0, 0.0
  total_images = 0

  for root, dirs, files in os.walk(dataset_path):
      for file in files:
          if file.endswith(('.jpg', '.jpeg', '.png')):
              image_path = os.path.join(root, file)
              img = Image.open(image_path)
              img_array = np.array(img) /255

              mean_r += np.mean(img_array[:, :, 0])
              mean_g += np.mean(img_array[:, :, 1])
              mean_b += np.mean(img_array[:, :, 2])

              total_images += 1

  mean_r /= total_images
  mean_g /= total_images
  mean_b /= total_images
  
  return mean_r, mean_g, mean_b

def compute_dataset_std(dataset_path, mean):
  stdTemp = np.array([0.,0.,0.])
  std = np.array([0.,0.,0.])
  total_images = 0
   
  for root, dirs, files in os.walk(dataset_path):
    for file in files:
      if file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(root, file)
        img = Image.open(image_path)
        img_array = np.array(img)/255
        for j in range(3):
          stdTemp[j] += ((img_array[:,:,j] - mean[j])**2).sum()/(img_array.shape[0]*img_array.shape[1])

        total_images += 1
  
  std = np.sqrt(stdTemp/total_images)

  return std

# Function to perform mean subtraction on an image
def mean_subtraction(img_array, mean_values, std):
    #print(img_array[:,:,0])
    img_array[:, :, 0] = ((img_array[:, :, 0]/255) - mean_values[0]) / std[0]
    img_array[:, :, 1] = ((img_array[:, :, 1]/255) - mean_values[1]) / std[1]
    img_array[:, :, 2] = ((img_array[:, :, 2]/255) - mean_values[2]) / std[2]
    return img_array

# Custom preprocessing function including mean subtraction
def custom_preprocess(img_array, mean_values, std):
  img_array = mean_subtraction(img_array, mean_values, std)
  # You can add other preprocessing steps here if needed
  return img_array

def data_augmentation(augment: bool, params, mean_values, std):
  """
  Generates augmented data.
  :param augment: Boolean, when true the data augmentation is done.
  :return: ImageDataGenerator object instance
  """
  rotation_range=0
  width_shift_range=0.
  height_shift_range=0.
  shear_range=0.
  zoom_range=0.
  horizontal_flip=False


  if 'rotation' in params['data_aug']:
      rotation_range = 20
  if 'wsr' in params['data_aug']:
      width_shift_range=0.2
  if 'hsr' in params['data_aug']:
      height_shift_range=0.2
  if 'sr' in params['data_aug']:
      shear_range=0.2
  if 'zr' in params['data_aug']:
      zoom_range=0.2
  if 'hf' in params['data_aug']:
      horizontal_flip=True

  if augment:
    if mean_values[0]==0:
      data_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode = 'nearest'
      )
    else:
      data_generator = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function = lambda x: custom_preprocess(x, mean_values= mean_values, std=std),
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode = 'nearest'
      )
        
    return data_generator

  else:
    if mean_values[0] == 0:
      return ImageDataGenerator(rescale=1./255)
    else:
      return ImageDataGenerator(#rescale=1./255,
                                preprocessing_function = lambda x: custom_preprocess(x, mean_values= mean_values, std = std),
                                )
    
def load_data(data_object, w, h, bs, directory):
    """
    Loads the image generator object with the images from the chosen directory to the var data.
    :param data_object: ImageDataGenerator object instance
    :param directory: dataset directory
    :return:
    """
    data = data_object.flow_from_directory(
        directory=directory,
        target_size=(w, h),
        batch_size=bs,
        class_mode='categorical',
        shuffle=True
    )

    return data


def get_optimizer(params):
    if params['optimizer'] == 'adam':
        #optimizer = Adam(learning_rate=float(params['lr']), beta_1=float(params['momentum']))
        optimizer = Adam()
    elif params['optimizer'] == 'adadelta':
        optimizer = Adadelta(learning_rate=float(params['lr']), rho=float(params['momentum']))
    elif params['optimizer'] == 'sgd':
        #optimizer = SGD(learning_rate=float(params['lr']), momentum=float(params['momentum']))
        optimizer = SGD()
    elif params['optimizer'] == 'rmsprop':
        #optimizer = RMSprop(learning_rate=float(params['lr']), rho=float(params['momentum']))
        optimizer = RMSprop()

    return optimizer


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        print(im.size)
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')

def conv_block(n_filters, kernel_size, stride, bn, dp, pool, padding, input=0):
  
  #if input !=0:
  x = Conv2D(n_filters, (kernel_size,kernel_size), strides = (stride, stride), padding=padding)(input)
  #else:
    #model.add(Conv2D(n_filters, (kernel_size,kernel_size), strides = (stride, stride), padding=padding))
  
  if bn == 'True':
   x = BatchNormalization()(x)
  
  x = Activation('relu')(x)
  
  if pool == 'max':
    x = MaxPooling2D(pool_size=2)(x)
  elif pool == 'avg':
    x = AveragePooling2D(pool_size=2)(x)

  if dp != 0:
    x = Dropout(dp)(x)

  return x
     


def build_model(params, input):
  
  #model = Sequential()
  # Input layer

  x = conv_block(int(params['n_filters_1']), int(params['kernel_size_1']), params['stride'], params['bn'], float(params['dropout']), params['pool'], params['padding'], input)

  if params['depth'] > 1:
    x = conv_block(int(params['n_filters_2']), int(params['kernel_size_2']), params['stride'], params['bn'], float(params['dropout']), params['pool'], params['padding'], x)

    if params['depth'] > 2:
      x = conv_block(int(params['n_filters_3']), int(params['kernel_size_3']), params['stride'], params['bn'], float(params['dropout']), params['pool'], params['padding'], x)
    
      if params['depth'] > 3: 
        x = conv_block(int(params['n_filters_4']), int(params['kernel_size_4']), params['stride'], params['bn'], float(params['dropout']), params['pool'], params['padding'], x)


  model = Model(inputs = input, outputs = x)
  if params['pool']!='none':
    # Create a new Sequential model
    model = Model(inputs = input, outputs = model.layers[-1].output)
    x = model.output

    # Remove the last layer
    #model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

  #model.add(MaxPooling2D(pool_size=2))
  x= GlobalAveragePooling2D()(x)

  x=Dense(int(params['neurons']))(x)
  #if params['bn'] == 'True':
  #  model.add(BatchNormalization())
  x = Activation('relu')(x)

  x = Dense(params['output'], activation='softmax')(x)
  model = Model(inputs = input, outputs = x)
  model.summary()

  return model


def prune_model(model, pruning_threshold=0.1):
  for layer in model.layers[:-1]:
    if isinstance(layer, Dense):
      weights = layer.get_weights()
      pruned_weights = [w * (np.abs(w) >= pruning_threshold) for w in weights]
      layer.set_weights(pruned_weights)

  # Count the number of parameters in the pruned model
  num_parameters_pruned = model.count_params()
  print('Num of params after pruning: ', num_parameters_pruned)

  return 

def visualize_weights(model):
  dense = 0
  for layer in model.layers[:-1]:
    if isinstance(layer, Dense):
      weights = layer.get_weights()[0]
      # Flatten the weights to a 1D array
      weights_flat = weights.flatten()

      # Plot the histogram of weight values
      plt.hist(weights_flat, bins=50, color='blue', edgecolor='black')
      plt.title('Histogram of Weight Values')
      plt.xlabel('Weight Value')
      plt.ylabel('Frequency')
      plt.savefig(f'weights_{dense}.png')
      dense+=1
      