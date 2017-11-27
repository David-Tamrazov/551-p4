import keras
import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam as Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

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
BATCH_SIZE = 100


# function to fetch data - if testing = true, it'll fetch the entire dataset. otherwise it'll load the first 1000 lines
def fetch_data(fashion_mnist=True, mnist=True, testing=False):

    # get only fashion mnit 
    if fashion_mnist:
    
        # load fashion mnist from file 
        f_train_X, f_train_Y = load_file(fmnist_train_path, testing)
        f_test_X, f_test_Y = load_file(fmnist_test_path, testing)
        
        X_train = f_train_X
        Y_train = f_train_Y
        X_test = f_test_X
        Y_test = f_test_Y

    # get mnist data
    if mnist:
    
        # load mnist from file 
        m_train_X, m_train_Y = load_file(mnist_train_path, testing)
        m_test_X, m_test_Y = load_file(mnist_test_path, testing)
        
        X_train = m_train_X
        Y_train = m_train_Y
        X_test = m_test_X
        Y_test = m_test_Y
    
   
    # get both mnist and fashion mnist
    if mnist and fashion_mnist: 
        
        # concatenate fashion mnist and mnist together
        X_train = np.concatenate((m_train_X, f_train_X), axis=0)
        Y_train = np.concatenate((m_train_Y, f_train_Y), axis=0)
        X_test = np.concatenate((m_test_X, f_test_X), axis=0)
        Y_test = np.concatenate((m_test_Y, f_test_Y), axis=0)

    # convert to categorical one hot vectors
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
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



# callback function to anneal the learning rate
def lr_scheduler(epoch):
    
    # change the learning rate at the 26th epoch onwards
    if epoch > 25:
        return 1e-5
    
    return 1e-3


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
    model.add(Conv2D(10, (1, 1), input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1], strides = 2, activation = 'relu'))
    
    # Flatten
    model.add(Flatten())
    
    # add the final output softmax layer
    model.add(Dense(10, activation ='softmax'))
    
    # set the model parameters
    model.compile(Adam(lr = LEARNING_RATE), loss = 'categorical_crossentropy', metrics=['accuracy'])
    
    return model


def main():
    
    # fetch the data - MNIST only 
    X_train, Y_train, X_test, Y_test = fetch_data(fashion_mnist = False)

    # # fetch the data - fashion MNIST only 
    # X_train, Y_train, X_test, Y_test = fetch_data(mnist = False)

    # build the model 
    model = create_CNN_model()

    # callback for annealing the learning rate after 25 epochs
    callback = LearningRateScheduler(lr_scheduler)

    # run training
    model.fit(X_train, Y_train, 
            validation_data = (X_test, Y_test),
            epochs = EPOCHS, 
            batch_size = BATCH_SIZE, 
            callbacks=[callback],
            verbose = 2)



main()



