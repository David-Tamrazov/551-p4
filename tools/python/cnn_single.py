# example run:   python cnn_multi.py --train fm --test=f --learning_rate=0.009 --epoch=2 --mini_testing_data

#Put this code section before importing keras -- to save time when arguments were wrong
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--is_fashion", action="store_true", dest="is_fashion")
parser.add_argument("--is_not_mnist", action="store_true", dest="is_not_mnist")
parser.add_argument("--epoch", action="store", dest="epoch", type=int, default=25)
parser.add_argument("--load_multi", action="store_true", dest="load_multi")
parser.add_argument("--fresh", action="store_true", dest="fresh")
parser.add_argument("--mini_testing_data", action="store_false", dest="all_testing_data")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", type=float, default=0.001)
args = parser.parse_args()
print "Parsed arguments: " + str(args)


import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import Adam as Adam
from keras.utils import to_categorical

freeze_bottom = False

# filepaths to the fashion mnist data
fmnist_train_path = '../../../data/fashion_train.ocv'
fmnist_test_path = '../../../data/fashion_test.ocv'

# filepaths to the not mnist data
nmnist_train_path = '../../../data/not_mnist_train.ocv'
nmnist_test_path = '../../../data/not_mnist_test.ocv'

# filepaths to the mnist data
mnist_train_path = '../../../data/mnist_train.ocv'
mnist_test_path = '../../../data/mnist_test.ocv'

# filepath for the pre-trained multitask model
serialized_multitask_model_path = "./multitask_model.json"
serialized_multitask_weights_path = "./multitask_weights.h5"
serialized_mnist_single_model_path = "./single_model_1.json"
serialized_mnist_single_weights_path = "./single_weights_1.h5"
serialized_fashion_single_model_path = "./single_model_2.json"
serialized_fashion_single_weights_path = "./single_weights_2.h5"
serialized_not_mnist_single_model_path = "./single_model_3.json"
serialized_not_mnist_single_weights_path = "./single_weights_3.h5"

# hyper parameters 
IMAGE_SIZE = 28
EPOCHS = args.epoch
LEARNING_RATE = args.learning_rate
DECAY_RATE = LEARNING_RATE / EPOCHS
BATCH_SIZE = 100


# function to fetch data - if testing = true, it'll fetch the entire dataset. otherwise it'll load the first 1000 lines
def fetch_data(is_fashion, is_not_mnist, testing):
    num_classes = 10
    
    # get only fashion mnist 
    if is_fashion:
        print ">>    Fetching fashionMNIST dataset" 
    
        # load fashion mnist from file 
        f_train_X, f_train_Y = load_file(fmnist_train_path, testing)
        f_test_X, f_test_Y = load_file(fmnist_test_path, testing)

        X_train = f_train_X
        Y_train = f_train_Y
        X_test = f_test_X
        Y_test = f_test_Y

        print ">>    Adjusting fashionMNIST labels"
        Y_train -= 10
        Y_test -= 10

    # get only not mnist 
    elif is_not_mnist:
        print ">>    Fetching notMNIST dataset" 
    
        # load not mnist from file 
        n_train_X, n_train_Y = load_file(nmnist_train_path, testing)
        n_test_X, n_test_Y = load_file(nmnist_test_path, testing)

        X_train = n_train_X
        Y_train = n_train_Y
        X_test = n_test_X
        Y_test = n_test_Y

        print ">>    Adjusting notMNIST labels"
        Y_train -= 20
        Y_test -= 20

    # get mnist data
    else:
        print ">>    Fetching MNIST dataset"
    
        # load mnist from file 
        m_train_X, m_train_Y = load_file(mnist_train_path, testing)
        m_test_X, m_test_Y = load_file(mnist_test_path, testing)
        
        X_train = m_train_X
        Y_train = m_train_Y 
        X_test = m_test_X
        Y_test = m_test_Y 

    # convert to categorical one hot vectors
    Y_train = to_categorical(Y_train,num_classes=num_classes)
    Y_test = to_categorical(Y_test,num_classes=num_classes)
    
    return X_train, Y_train, X_test, Y_test

# Loads and returns the image data found at the filepath 
def load_file(filepath, testing):
    
    # read the image file 
    if testing:
        tmp = pd.read_csv(filepath, sep=' ', skiprows=1).values; 
    else:
        tmp = pd.read_csv(filepath, sep=' ', nrows=10, skiprows=1).values; 
    
    # split the data between pixel and meta 
    meta_data = tmp[:, 0:2]
    pixel_data = tmp[:, 2:]
    
    # reshape the pixel data into 28x28x1
    X = np.reshape(pixel_data, (-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    # extract labels from the meta data
    Y = meta_data[:,0:1]

    return X, Y


# outlined this function to compile all models with the same parameter
# TODO may need to modify according to the paper
def compile_model(model):
    model.compile(Adam(lr = LEARNING_RATE, decay=DECAY_RATE), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model
    
def create_CNN_model():
    
    # initialize a sequential model
    model = Sequential()
    
    # build the architecture
    model.add(Conv2D(96, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 1, activation ='relu'))
    model.add(Conv2D(96, (3, 3), strides = 1, activation = 'relu'))
    model.add(Conv2D(96, (3, 3), strides = 2, activation = 'relu'))
    
    model.add(Conv2D(192, (3, 3) , strides = 1, activation ='relu'))
    model.add(Conv2D(192, (3, 3), strides = 1, activation = 'relu'))
    model.add(Conv2D(192, (3, 3), strides = 2, activation = 'relu'))
    
    model.add(Conv2D(192, (3, 3) , strides = 1, activation ='relu'))
    model.add(Conv2D(192, (1, 1), strides = 1, activation = 'relu'))
    model.add(Conv2D(10, (1, 1), strides = 2, activation = 'relu'))
    
    # Flatten
    model.add(Flatten())
    
    # add the final output softmax layer
    model.add(Dense(10, activation ='softmax'))
    
    # set the model parameters
    compile_model(model)
    
    return model

def save_pretrained_model(model, is_fashion, is_not_mnist):
    if is_fashion:
        model_save_path = serialized_fashion_single_model_path 
        weights_save_path = serialized_fashion_single_weights_path
    elif is_not_mnist:
        model_save_path = serialized_not_mnist_single_model_path
        weights_save_path = serialized_not_mnist_single_weights_path
    else:
        model_save_path = serialized_mnist_single_model_path
        weights_save_path = serialized_mnist_single_weights_path

    # save the model
    print ">>   JSON path: {}".format(model_save_path)
    json_file = open(model_save_path, "w")
    model_json = model.to_json()
    json_file.write(model_json)
    json_file.close()
    
    # save the pre-trained weights
    print ">>    Weights path: {}".format(weights_save_path)
    model.save_weights(weights_save_path)


def load_pretrained_model(model_file, weights_file):
    
    print ">>    Loading JSON model: {}".format(model_file)
    json_file = open(model_file, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    print ">>    Loading weights: {}".format(weights_file)
    loaded_model.load_weights(weights_file)
    compile_model(loaded_model)

    return loaded_model

def convert_to_single_task(model):
    
    print ">>    Popping top 3 layers"
    # remove the top 3 layers - 20-class softmax, flatten, 20-filter conv 
    model.pop()
    model.pop()
    model.pop()

    # add the final output softmax layer
    # fix all the convolutional layers, only train the fully connected layer on top
    # TODO should we consider fix-train-unfix-train?
    if freeze_bottom:
        for layer in loaded_model.layers:
            layer.trainable = False

    # add 10-class versions of the same layers 
    print ">>    Adding 10-class versions of the same layers"
    model.add(Conv2D(10, (1, 1), strides = 2, activation = 'relu', name="NEW_TOP"))
    model.add(Flatten(name="NEW_FLATTEN"))
    model.add(Dense(10, activation ='softmax', name="NEW_SOFTMAX"))

    # compile and return the model 
    return compile_model(model)

# Load model
def load_model(load_multi, is_fashion, is_not_mnist, fresh):
    if load_multi:
        print ">> Loading multi-task model"
        model = load_pretrained_model(serialized_multitask_model_path, serialized_multitask_weights_path)
        model = convert_to_single_task(model)
    elif fresh:
        print ">> Creating a new single-task model"
        model = create_CNN_model()
    elif is_fashion:
        print ">> Loading fashionMNIST single-task model"
        model = load_pretrained_model(serialized_fashion_single_model_path, serialized_fashion_single_weights_path)
    elif is_not_mnist:
        print ">> Loading notMNIST single-task model"
        model = load_pretrained_model(serialized_not_mnist_single_model_path, serialized_not_mnist_single_weights_path)
    else:
        print ">> Loading MNIST single-task model"
        model = load_pretrained_model(serialized_mnist_single_model_path, serialized_mnist_single_weights_path)
    return model

def main():
    # fetch the data - MNIST only 
    print ">> Fetching data ..."
    X_train, Y_train, X_test, Y_test = fetch_data(args.is_fashion, args.is_not_mnist, args.all_testing_data)

    # X_train, Y_train, X_test, Y_test = fetch_data(mnist = False)

    # build the multitask_model 
    model = load_model(args.load_multi, args.is_fashion, args.is_not_mnist, args.fresh)

    # run training
    print ">> Start training"
    model.fit(X_train, Y_train, 
                        validation_data = (X_test, Y_test),
                        epochs = args.epoch, 
                        batch_size = BATCH_SIZE, 
                        callbacks=[],
                        verbose = 2)
    print ">> Training completed!"

    print ">> Saving model to disk"
    save_pretrained_model(model,args.is_fashion, args.is_not_mnist)
    print ">> Model saved to disk"
    print "Done"

main()

#def run_test_on(name):
#    # load model for mnist
#    # load model for mnist
#    mn_model = load_pretrained_model()
#
#    # convert the model to single-task learner 
#    mn_model = convert_to_single_task(mn_model)
#
#    X_train, Y_train, X_test, Y_test = fetch_data(mnist = ("m" == name), fashion_mnist = ("f" == name), testing = args.all_testing_data)
#    mn_model.fit(X_train, Y_train, 
#            validation_data = (X_test, Y_test),
#            epochs = args.epoch, 
#            batch_size = BATCH_SIZE, 
#            callbacks=[],
#            verbose = 2)
