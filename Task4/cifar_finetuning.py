from utils import *
import keras
import tensorflow as tf
from keras.utils import plot_model
from keras.datasets import cifar10

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
#import tensorflow_model_optimization 

import tensorflow as tf

import optuna
import wandb

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_large_train'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_TEST = '/ghome/mcv/datasets/C3/MIT_large_train'

# Put your key
wandb.login(key='4c0b25a1f87331e99edadbaa2cf9568a452224ff') #GOIO
#wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')  # IKER


# wandb.login(key='50315889c64d6cfeba1b57dc714112418a50e134') #Xavi


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
    NUMBER_OF_EPOCHS = params['epochs']
    CIFAR10_MODEL_FNAME = '/ghome/group06/lab4_C3_G06/weights/cifar10_earlystop.keras'
    BEST_MODEL_FNAME = f'/ghome/group06/lab4_C3_G06/weights/finetunedCifar10_224.keras'
    validation_split = 0.2
    # Calculate steps per epoch
    steps_per_epoch = 400 // BATCH_SIZE

    #(input_train, target_train), (input_test, target_test) = cifar10.load_data()
    
    mean_r, mean_g, mean_b = (0,0,0)

    if params['substract_mean'] == 'True':
        # Compute the mean values of the dataset
        mean_r, mean_g, mean_b = compute_dataset_mean(DATASET_DIR+'/train/')
        std = compute_dataset_std(DATASET_DIR+'/train/', (mean_r, mean_g, mean_b))
        print(mean_r, mean_g, mean_b)
        print(std)

    # # Parse numbers as floats
    # input_train = input_train.astype('float32')
    # input_test = input_test.astype('float32')

    # # Normalize data
    # input_train = input_train / 255
    # input_test = input_test / 255

    # Train data
    train_data_generator = data_augmentation(True, params, (mean_r, mean_g, mean_b), std)
    train_dataset = load_data(train_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_DIR + '/train/')

    # Validation data
    validation_data_generator = data_augmentation(False, params, (mean_r, mean_g, mean_b), std)
    validation_dataset = load_data(validation_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_DIR + '/test/')

    # Test Data
    test_data_generator = data_augmentation(False, params, (mean_r, mean_g, mean_b), std)
    test_dataset = load_data(test_data_generator, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, directory=DATASET_TEST + '/test/')

    base_model = build_model(params, input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    base_model.load_weights(CIFAR10_MODEL_FNAME)
    plot_model(base_model, to_file='/ghome/group06/lab4_C3_G06/model_cifar', show_shapes=True, show_layer_names=True)
    exit()

    model = Sequential()
    for layer in base_model.layers[:-1]:
        model.add(layer)
    model.add(Dense(8, activation='softmax'))

    model.summary()

    #Plot model
    # Get the total number of parameters
    total_parameters = model.count_params()

    # Create the ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
                                factor=0.1,          # Reduce the learning rate by a factor of 0.1
                                patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
                                min_lr=1e6)         # Minimum learning rate

    earlystop = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1,
                              baseline=None, restore_best_weights=True)

    checkpoint = ModelCheckpoint(BEST_MODEL_FNAME,  # Filepath to save the model weights
                                 monitor='val_accuracy',  # Metric to monitor for saving the best model
                                 save_best_only=True,  # Save only the best model (based on the monitored metric)
                                 mode='max', # 'max' or 'min' depending on whether the monitored metric should be maximized or minimized
                                 verbose=1)  # Verbosity mode, 1 for progress updates

    callbacks = [reduce_lr, checkpoint]

    #Get optimizer
    optim = get_optimizer(params)

    for layer in model.layers[:-2]:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    
    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        batch_size = BATCH_SIZE,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_dataset,
                        verbose=2,
                        callbacks=callbacks)
    
    # Load the best weights before calling model.evaluate
    model.load_weights(BEST_MODEL_FNAME)
    #visualize_weights(model)

    score = model.evaluate(test_dataset, verbose=0)
    ratio = score[1]/(total_parameters/100000)

    print(score)
    print(history.history.keys())
    print(ratio)

    

    # # Define the pruning parameters
    # pruning_thr = params['pruning_thr']

    # prune_model(model, pruning_thr)
    # visualize_weights(model)
    # # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    # # train the model on the new data for a few epochs
    # history = model.fit(train_dataset,
    #                     epochs=NUMBER_OF_EPOCHS,
    #                     validation_data=validation_dataset,
    #                     verbose=2,
    #                     callbacks=callbacks)

    # # Load the best weights before calling model.evaluate
    # model.load_weights(BEST_MODEL_FNAME)

    # result = model.evaluate(test_dataset, verbose=0)
    # ratio = result[1]/(total_parameters/100000)

    # print(result)
    # print(history.history.keys())
    # print(ratio)

    return history, score, total_parameters, ratio


def objective_model_cv(trial):
    
    params = {
        #'unfrozen_layers': trial.suggest_categorical('unfrozen_layers', ["1"]),  # 1,2,3,4,5
        
        'substract_mean': trial.suggest_categorical('substract_mean', ['True']),
        'batch_size': trial.suggest_categorical('batch_size', ['16']),  # 8,16,32,64
        'img_size': trial.suggest_categorical('img_size', ['224']),  # 8,16,32,64,128,224,256
        'lr': trial.suggest_float('lr', 1., 1.),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        'optimizer': trial.suggest_categorical('optimizer', ['adadelta']),  # adadelta, adam, sgd, RMSprop


        'activation': trial.suggest_categorical('activation', ['relu']),
        'n_filters_1': trial.suggest_categorical('n_filters_1', ['64']),
        'n_filters_2': trial.suggest_categorical('n_filters_2', ['16']),
        'n_filters_3': trial.suggest_categorical('n_filters_3', ['64']),
        'n_filters_4': trial.suggest_categorical('n_filters_4', ['32']),

        'kernel_size_1': trial.suggest_categorical('kernel_size_1', ['3']),
        'kernel_size_2': trial.suggest_categorical('kernel_size_2', ['3']),
        'kernel_size_3': trial.suggest_categorical('kernel_size_3', ['3']),
        'kernel_size_4': trial.suggest_categorical('kernel_size_4', ['3']),

        'stride': trial.suggest_int('stride', 1,1),

        'pool': trial.suggest_categorical('pool', ['max']),

        'padding': trial.suggest_categorical('padding', ['same']),
        'neurons': trial.suggest_categorical('neurons', ['256']),

        'data_aug': trial.suggest_categorical('data_aug', ['sr']),
        'momentum': trial.suggest_float('momentum', 0.95, 0.95),
        'dropout': trial.suggest_categorical('dropout', ['0']),
        'bn': trial.suggest_categorical('bn', ['True']),
        'L2': trial.suggest_categorical('L2', ['False']),
        'epochs': trial.suggest_int('epochs', 100,100),
        'depth': trial.suggest_int('layer_n', 4,4), 
        'pruning_thr': trial.suggest_float('pruning_thr', 0.1, 0.1),
        'output': trial.suggest_int('output', 10,10),
    }
    
    config = dict(trial.params)
    config['trial.number'] = trial.number

    execution_name = 'Lab4_test_'+str(params['optimizer'])

    wandb.init(
        project='Lab4_optuna_cifar10',
        entity='c3_mcv',
        name=execution_name,
        config=config,
        reinit=True,
    )
    history, result, total_parameters,ratio = model(params, trial.number)

    # report validation accuracy to wandb
    for epoch in range(len(history.history['accuracy'])):
        wandb.log({
            'Train Loss': history.history['loss'][epoch],
            'Validation Loss': history.history['val_loss'][epoch],
            'Train Accuracy': history.history['accuracy'][epoch],
            'Validation Accuracy': history.history['val_accuracy'][epoch],
        })

    wandb.log(data={"Test Accuracy": result[1]})
    wandb.log(data={"Test Loss": result[0]})
    wandb.log(data={"Total Parameters": total_parameters})
    wandb.log(data={"Ratio": ratio})
    
    return ratio


study = optuna.create_study(direction="maximize", study_name='Lab_4')
study.optimize(objective_model_cv, n_trials=1)
