from utils import *
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, Adadelta, RMSprop

# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# Import DenseNet121
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input

from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import cv2
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_large_train'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_TEST = '/ghome/mcv/datasets/C3/MIT_large_train'

# Put your key
# wandb.login(key='4c0b25a1f87331e99edadbaa2cf9568a452224ff') #GOIO
#wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')  # IKER


# wandb.login(key='50315889c64d6cfeba1b57dc714112418a50e134') #Xavi
def confusion(actual, predicted, classes):
    conf = np.zeros((len(classes), len(classes)), dtype=np.float64)

    mp = {}
    for i, cls in enumerate(classes):
        mp[cls] = i

    for a, p in zip(actual, predicted):
        conf[a, p] += 1

    print(conf)
    #conf = conf / conf.sum(axis=1, keepdims=True)
    ax = sns.heatmap(conf, cmap='YlGnBu', xticklabels=classes, yticklabels=classes, square=True, annot=True)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title('Confusion Matrix')
    plt.savefig('visualizations/conf_map.png')
    return 




def visualizate_save(model, IMG_SIZE):
    directory_test = DATASET_TEST+'/test'
    class_nodict = ['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding']
    classes = {'Opencountry':0, 'coast':1,'forest':2,'highway':3,'inside_city':4,'mountain':5,'street':6,'tallbuilding':7}
    classes_inv = list(classes.keys())

    test_labels = []
    preds = []
    mean_rgb = (0.4323919788435387, 0.4514907637951422, 0.44213367058249065)
    std = (0.25540545, 0.24442602, 0.27033866)

    for class_dir in os.listdir(directory_test):
        cls = classes[class_dir]
        for imname in os.listdir(os.path.join(directory_test,class_dir)):
            im = mean_subtraction(get_img_array(os.path.join(directory_test,class_dir,imname), size=IMG_SIZE), mean_rgb, std)
            pred = model.predict(im)
            out = np.argmax(pred[0])
            # im = preprocess_input(get_img_array(os.path.join(directory_test, class_dir, imname), size=IMG_SIZE))
            # pred = model.predict(im)
            # out = np.argmax(pred[0])

            # icam = GradCAM(model, out, 'conv2d_1') 
            # heatmap = icam.compute_heatmap(im)
            # heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

            # image = cv2.imread(os.path.join(directory_test,class_dir,imname))
            # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            # (heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)


            # if out==cls:
            #     cv2.imwrite(f'visualizations/correct/{class_dir}/heatmap_{imname}', heatmap) 
            #     cv2.imwrite(f'visualizations/correct/{class_dir}/output_{imname}', output) 
            # else: 
            #     cv2.imwrite(f'visualizations/incorrect/{class_dir}/heatmap_{classes_inv[out]}_{imname}', heatmap) 
            #     cv2.imwrite(f'visualizations/incorrect/{class_dir}/output_{classes_inv[out]}_{imname}', output) 
            
            test_labels.append(cls)
            preds.append(out)
    
    confusion(test_labels, preds, class_nodict)

    return

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        
        gradModel = Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

def decode_custom_predictions(predictions, custom_class_labels):
    # Assuming predictions is a vector of probabilities for each class
    # custom_class_labels is a list of your custom class labels

    # Get the index with the highest probability
    top_prediction_index = np.argmax(predictions)

    # Get the corresponding custom class label and probability
    top_custom_class_label = custom_class_labels[top_prediction_index]
    top_probability = predictions[top_prediction_index]

    return top_custom_class_label, top_probability, top_prediction_index

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


print(f'gpus? {keras.distribution.list_devices(device_type="GPU")}')
print('GPU name: ', tf.config.list_physical_devices('GPU'))


def model(params):
    """
    CNN model configuration
    """
    # Define the test data generator for data augmentation and preprocessing
    IMG_WIDTH = int(params['img_size'])
    IMG_HEIGHT = int(params['img_size'])

    BEST_MODEL_FNAME = f'/ghome/group06/lab4_C3_G06/weights/best_weights_64_16.keras'

    # create the base pre-trained model
    # base_model = VGG16(weights='imagenet', include_top=False)
    input = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model = build_model(params, input)

    for layer in model.layers[:-1]:
        print(layer.name)

    optim = get_optimizer(params)

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the best weights
    model.load_weights(BEST_MODEL_FNAME)
    out = model(input)
    
    visualizate_save(model, IMG_WIDTH)
    

params = {
    #'unfrozen_layers': trial.suggest_categorical('unfrozen_layers', ["1"]),  # 1,2,3,4,5
    
    'substract_mean': 'True',
    'batch_size': '16',  # 8,16,32,64
    'img_size': '224',  # 8,16,32,64,128,224,256
    'lr': 1,  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
    'optimizer': 'adadelta',  # adadelta, adam, sgd, RMSprop


    'activation': 'relu',
    'n_filters_1': '64',
    'n_filters_2': '16',
    'n_filters_3': '16',
    'n_filters_4': '16',

    'kernel_size_1': '3',
    'kernel_size_2': '3',
    'kernel_size_3': '3',
    'kernel_size_4': '3',

    'stride': 1,

    'pool': 'max',

    'padding': 'same',
    'neurons': '256',

    'data_aug': 'sr',
    'momentum': 0.95,
    'dropout': '0',
    'bn': 'True',
    'L2': 'False',
    'epochs': 100,
    'depth': 2, 
    'pruning_thr': 0.1,
    'output': 8,
}

#Execute the 'main'
model(params)
