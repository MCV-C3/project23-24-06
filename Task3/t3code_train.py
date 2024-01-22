from utils import *
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.regularizers import l2

from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import optuna
import wandb

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_large_train'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_TEST = '/ghome/mcv/datasets/C3/MIT_large_train'

# Put your key
# wandb.login(key='4c0b25a1f87331e99edadbaa2cf9568a452224ff') #GOIO
wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')  # IKER


# wandb.login(key='50315889c64d6cfeba1b57dc714112418a50e134') #Xavi

def truncate(params, model):
    #Truncate model on the specified layer
    layer_n = params['layer_n']
    truncated_model = Model(inputs = model.input, outputs = model.layers[-layer_n].output)

    return truncated_model

def get_optimizer(params):
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=float(params['lr']), beta_1=float(params['momentum']))
    elif params['optimizer'] == 'adadelta':
        optimizer = Adadelta(learning_rate=float(params['lr']), rho=float(params['momentum']))
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=float(params['lr']), momentum=float(params['momentum']))
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=float(params['lr']), rho=float(params['momentum']))

    return optimizer

def data_augmentation(augment: bool, params):
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
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input,
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
        return ImageDataGenerator(preprocessing_function=preprocess_input)


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


print(f'gpus? {keras.distribution.list_devices(device_type="GPU")}')
print('GPU name: ', tf.config.list_physical_devices('GPU'))


def model(params, number):
    """
    CNN model configuration
    """
    # Define the test data generator for data augmentation and preprocessing
    IMG_WIDTH = int(params['img_size'])
    IMG_HEIGHT = int(params['img_size'])
    BATCH_SIZE = int(params['batch_size'])
    NUMBER_OF_EPOCHS_WU = 20
    NUMBER_OF_EPOCHS = params['epochs']
    BEST_MODEL_FNAME = f'/ghome/group06/lab3_C3_G06/weights/best_weights_L2.keras'

    # Train data
    train_data_generator = data_augmentation(True, params)
    train_dataset = load_data(train_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_DIR + '/train/')

    # Validation data
    validation_data_generator = data_augmentation(False, params)
    validation_dataset = load_data(validation_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_DIR + '/test/')

    # Test Data
    test_data_generator = data_augmentation(False, params)
    test_dataset = load_data(test_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_TEST + '/test/')

    # create the base pre-trained model
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    #Truncate the model on the specified layer
    if params['layer_n'] > 0:
        base_model = truncate(params, base_model)

    # add a global spatial average pooling layer    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # let's add a fully-connected layer
    if params['L2'] == 'True':
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    else:
        x = Dense(1024, activation='relu')(x)


    if params['batch_norm'] == 'True':
        x = BatchNormalization()(x)
    if params['dropout'] > 0:
        x = Dropout(params['dropout'])(x)

    # and a classification layer for 8 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # Warmup
    for layer in model.layers[:-3]:
        layer.trainable = False
        print(layer.name)

    #Plot model
    plot_model(model, to_file='modelDenseNet121a.png', show_shapes=True, show_layer_names=True)

    #earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1,
    #                          baseline=None, restore_best_weights=True)

    checkpoint = ModelCheckpoint(BEST_MODEL_FNAME,  # Filepath to save the model weights
                                 monitor='val_accuracy',  # Metric to monitor for saving the best model
                                 save_best_only=True,  # Save only the best model (based on the monitored metric)
                                 mode='max', # 'max' or 'min' depending on whether the monitored metric should be maximized or minimized
                                 verbose=1)  # Verbosity mode, 1 for progress updates

    callbacks = [checkpoint]

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS_WU,
                        validation_data=validation_dataset,
                        verbose=2,
                        callbacks=callbacks)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from DenseNet121. We will freeze the bottom N layers
    # and train the remaining top layers.

    # we chose to train the top N layers, i.e. we will freeze
    # the first layers and unfreeze the N rest:

    #Unfroze 1 layer = 
    #Unfroze 2 layers = 117
    #Unfroze 3 layers = 289
    #Unfroze 4 layers = 377
    #Unfroze all layers

    N=289
    for layer in model.layers[:N]:
        layer.trainable = False
    for layer in model.layers[N:]:
        layer.trainable = True
    

    #Get optimizer
    optim = get_optimizer(params)

    # we need to recompile the model for these modifications to take effect
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_dataset,
                        verbose=2,
                        callbacks=callbacks)

    # Load the best weights before calling model.evaluate
    model.load_weights(BEST_MODEL_FNAME)

    result = model.evaluate(test_dataset, verbose=0)
    print(result)
    print(history.history.keys())

    # list all data in history

    if True:
        import matplotlib
        # matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('accuracy.jpg')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('loss.jpg')

    return history, result


def objective_model_cv(trial):
    
    params = {
        #'unfrozen_layers': trial.suggest_categorical('unfrozen_layers', ["1"]),  # 1,2,3,4,5
        'batch_size': trial.suggest_categorical('batch_size', ['8','16','32','64']),  # 8,16,32,64
        'img_size': trial.suggest_categorical('img_size', ['224']),  # 8,16,32,64,128,224,256
        'lr': trial.suggest_float('lr', 0.0001, 0.3),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        'optimizer': trial.suggest_categorical('optimizer', ['adadelta', 'adam', 'sgd', 'rmsprop']),  # adadelta, adam, sgd, RMSprop
        'data_aug': trial.suggest_categorical('data_aug', ['hf']),
        'momentum': trial.suggest_float('momentum', 0, 1.0),
        'dropout': trial.suggest_float('dropout', 0, 1.0),
        'batch_norm': trial.suggest_categorical('batch_norm', ['True', 'False']),
        'L2': trial.suggest_categorical('L2', ['True', 'False']),
        'epochs': trial.suggest_int('epochs', 50,50),
        'layer_n': trial.suggest_int('layer_n', 0,0), 
    }
    
    config = dict(trial.params)
    config['trial.number'] = trial.number

    execution_name = 'Lab3_1dense_block_out_'+str(params['layer_n'])

    wandb.init(
        project='Lab3-layer_out',
        entity='c3_mcv',
        name=execution_name,
        config=config,
        reinit=True,
    )
    history, result = model(params, trial.number)

    # report validation accuracy to wandb
    for epoch in range(len(history.history['accuracy'])):
        wandb.log({
            'Epoch': epoch,
            'Train Loss': history.history['loss'][epoch],
            'Validation Loss': history.history['val_loss'][epoch],
            'Train Accuracy': history.history['accuracy'][epoch],
            'Validation Accuracy': history.history['val_accuracy'][epoch],
        })

    wandb.log(data={"Test Accuracy": result[1]})
    wandb.log(data={"Test Loss": result[0]})
    
    return result[1]


study = optuna.create_study(direction="maximize", study_name='Lab_3')
study.optimize(objective_model_cv, n_trials=1)
