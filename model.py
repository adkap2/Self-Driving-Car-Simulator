import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
from keras.models import load_model
import argparse
import os


def plot_initial_steering_angle(data):
    """ Plots the initial steering angle from data generated"""
    num_bins = 25
    samples_per_bin = 200
    hist, bins = np.histogram(data['steering'], num_bins)
    center = bins[:-1] + bins[1:] * 0.5  # center the bins to 0
    plt.bar(center, hist, width=0.05)
    #plt.plot((np.min(data['steering']),
    #  np.max(data['steering'])), (samples_per_bin, samples_per_bin))
    plt.title("Distribution of input steering angles")
    plt.xlabel("Normalized Steering Angles")
    plt.ylabel("Frequency")
    plt.show()


def plot_train_val_steering(y_train, y_valid):
    """ Plots the steering angles for the training and validation sets"""

    num_bins = 25
    samples_per_bin = 200
    print("Training Samples: {}\nValid Samples: {}".
    format(len(y_train), len(y_valid)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    axes[0].set_title('Training set')
    axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    axes[1].set_title('Validation set')
    plt.show()

# Initializes random seed
np.random.seed(100)

def load_data(args):
    """ Loads data in from driving log
    returns training a validation data
    """
    # Load data from driving log csv file
    # Specifies column names
    data = pd.read_csv(os.path.join(os.getcwd(),
     args.data_dir, 'driving_log.csv'),
     names=['center', 'left', 'right', 'steering',
      'throttle', 'reverse', 'speed'])
    X = data[['center', 'left', 'right']].values
    y = data['steering'].values
    #plot_initial_steering_angle(data)
    # Splits data to training and validation data at 80/20 split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
     test_size=0.2, random_state=0)
    #plot_train_val_steering(y_train, y_valid)

    return X_train, X_valid, y_train, y_valid

def build_model(args):
    """ Builds tensorflow Convolutional 2D model
    returns Model
    """
    model = Sequential()    # Sequential Model used
    # Initial input layer of shape equal to the input shape defined in utils.py
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    # Creates convolutional 2D layer with a filter size of 24 and a 5x5 kernal
    # Uses elu activation function to avoid vanishing gradient
    #model.add(Conv2D(24, (5, 5),  strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    # # Multiple convolutional layers used to generate deep neural network
    #model.add(Conv2D(48, (5, 5), strides=(2, 2),activation='elu'))
    #model.add(Conv2D(64, (3, 3), activation='elu'))
    # # Multiple model dropouts added at 50% drop to eliminate overfitting
    model.add(Flatten())
    #model.add(Dense(100, activation='elu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """ Trains CNN model using the model.fit function
    saves model to directory"""
    print(model.summary())
    # Compiles model with Adam optimizer and meansquared
    #  error loss with accuracy metrics
    model.compile(loss='mse', optimizer=Adam
    (lr=args.learning_rate), metrics=['accuracy'])
    # model.fit function fits training and validation data to model
    #  with default batch size, learning rate and number of epochs
    # Uses batch generator function to pull image and steering data
    #  as this way less memory is used for each iteration
    history = model.fit(batch_generator(args.data_dir,
     X_train, y_train, 800, True),
     epochs=args.nb_epoch, validation_data=
     batch_generator(args.data_dir, X_valid, y_valid, 800, True),
      batch_size=args.batch_size, verbose=1, shuffle=1,
       steps_per_epoch = args.samples_per_epoch)
    # Saves model to h5 file
    model.save('model', save_format='h5')


def main():
    """ Main function initializes default hyperparameter arguments
    calls load data function, build model function and train_model function
    """
    parser = argparse.ArgumentParser(description=
    'Behaivioral Cloning Training Program')
    parser.add_argument('-d', help = 'data directory',
     dest= 'data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction',
     dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='tdrop out probability',
     dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',
     dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='samples per epoch',
     dest='samples_per_epoch', type=int, default=40)
    parser.add_argument('-b', help='batch size',
     dest='batch_size', type=int, default=128)
    parser.add_argument('-l', help='learning rate',
     dest='learning_rate', type=float, default=1.0e-2)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')

    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)
    #model = load_model('model')

    # Evaluates model with validation data
    model.evaluate(batch_generator(args.data_dir, data[1], data[3], 1800, False))
    x_array, y_array = batch_generator(args.data_dir, data[1], data[3], 400, False)
    # Makes predictions on model
    predictions = model.predict(x_array, steps=args.samples_per_epoch)


if __name__=='__main__':
    main()
