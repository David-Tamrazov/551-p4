import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam as Adam

# filepaths to the fashion mnist data
fmnist_train_path = '../../data/fashion_ocv/fashion_train.ocv'
fmnist_test_path = '../../data/fashion_ocv/fashion_train.ocv'

# filepaths to the mnist data
mnist_train_path = '../../data/mnist_ocv/mnist_train.ocv'
mnist_test_path = '../../data/mnist_ocv/mnist_test.ocv'


# hyper parameters 
IMAGE_SIZE = 28
EPOCHS = 50
LEARNING_RATE = 1e-3
MINI_BATCH_SIZE = 100


# function to fetch data
def fetch_data(fashion_mnist=True, mnist=True, nu_lines=1000):

    # get both mnist and fashion mnist
    if fashion_mnist:
    
        # load fashion mnist from file 
        f_train_X, f_train_Y = load_file(fmnist_train_path, nu_lines)
        f_test_X, f_test_Y = load_file(fmnist_test_path, nu_lines)
        
        X_train = f_train_X
        Y_train = f_train_Y
        X_test = f_test_X
        Y_test = f_test_Y

    # get mnist data
    if mnist:
    
        # load mnist from file 
        m_train_X, m_train_Y = load_file(mnist_train_path, nu_lines)
        m_test_X, m_test_Y = load_file(mnist_test_path, nu_lines)
        
        X_train = m_train_X
        Y_train = m_train_Y
        X_test = m_test_X
        Y_test = m_test_Y
    
    # get only fashion mnit 
    if mnist and fashion_mnist: 
        
        # concatenate fashion mnist and mnist together
        X_train = np.concatenate((m_train_X, f_train_X), axis=0)
        Y_train = np.concatenate((m_train_Y, f_train_Y), axis=0)
        X_test = np.concatenate((m_test_X, f_test_X), axis=0)
        Y_test = np.concatenate((m_test_Y, f_test_Y), axis=0)
    
    return X_train, Y_train, X_test, Y_test



# Loads and returns the image data found at the filepath 
def load_file(filepath, nu_lines):
    
    # read the image file 
    tmp = pd.read_csv(filepath, sep=' ', nrows=nu_lines, skiprows=1).values; 
    
    # split the data between pixel and meta 
    meta_data = tmp[:, 0:2]
    pixel_data = tmp[:, 2:]
    
    # reshape the pixel data into 28x28x1
    X = np.reshape(pixel_data, (-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    
    # extract labels from the meta data
    Y = meta_data[:,0:1]

    return X, Y

def create_CNN_model():
    
    # initialize a sequential model
    model = Sequential()
    
    # build the architecture
    model.add(Conv2D(96, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1] , strides = 1, activation ='relu'))
    model.add(Conv2D(96, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 1, activation = 'relu'))
    model.add(Conv2D(96, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 2, activation = 'relu'))
    
    model.add(Conv2D(192, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1] , strides = 1, activation ='relu'))
    model.add(Conv2D(192, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 1, activation = 'relu'))
    model.add(Conv2D(192, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 2, activation = 'relu'))
    
    model.add(Conv2D(192, (3, 3), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1] , strides = 1, activation ='relu'))
    model.add(Conv2D(192, (1, 1), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 1, activation = 'relu'))
    model.add(Conv2D(20, (1, 1), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 2, activation = 'relu'))
    
    # Final output softmax layer
    model.add(Dense(20, activation ='softmax'))
    
    # set the model hyperparameters 
    model.compile(Adam(), loss = 'categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    
    # fetch the data
    X_train, Y_train, X_test, Y_test = fetch_data()

    # build the model 
    model = create_CNN_model()





