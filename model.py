import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from keras.optimizers import Adam
from keras.applications import ResNet50

from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from utils import INPUT_SHAPE, batch_generator

import argparse

import os



np.random.seed(100)

def load_data(args):

    data = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    X = data[['center', 'left', 'right']].values

    y = data['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state = 0)

    return X_train, X_valid, y_train, y_valid

def build_model(args):
    
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

    for layer in resnet.layers[:-4]:
        layer.trainable = False
    

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5),  strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2),activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    
    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=args.save_best_only,
                                mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        validation_steps=len(X_valid),
                        max_queue_size=1)
    
def s2b(s):

    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def main():

    parser = argparse.ArgumentParser(description='Behaivioral Cloning Training Program')
    parser.add_argument('-d', help = 'data directory', dest= 'data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='tdrop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')

    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)

if __name__=='__main__':
    main()
