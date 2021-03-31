import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

print(INPUT_SHAPE)

def plot_initial_steering_angle(data):
    num_bins = 25
    samples_per_bin = 200
    hist, bins = np.histogram(data['steering'], num_bins)
    center = bins[:-1] + bins[1:] * 0.5  # center the bins to 0

    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
    plt.show()

    
def plot_train_val_steering(y_train, y_valid):

    num_bins = 25
    samples_per_bin = 200

    print("Training Samples: {}\nValid Samples: {}".format(len(y_train), len(y_valid)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    axes[0].set_title('Training set')
    axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    axes[1].set_title('Validation set')
    plt.show()


np.random.seed(100)

def load_data(args):

    data = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    X = data[['center', 'left', 'right']].values

    y = data['steering'].values
    plot_initial_steering_angle(data)

    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state = 0)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    plot_train_val_steering(y_train, y_valid)


    


    return X_train, X_valid, y_train, y_valid

def build_model(args):
    
    # resnet = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    # for layer in resnet.layers[:-4]:
    #     layer.trainable = False
    

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
    model.add(Dropout(args.keep_prob))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(1))
    model.summary()

    # from keras.applications import ResNet50
    # resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

    # for layer in resnet.layers[:-4]:
    #     layer.trainable = False
    

    # model = Sequential()
    # model.add(resnet)
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.summary()
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=INPUT_SHAPE))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='linear'))

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):

    # print(f"This is the {len(X_train)}")
    # print(X_train.shape)
    print(model.summary())

    # checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
    #                             monitor='val_loss',
    #                             verbose=0,
    #                             save_best_only=args.save_best_only,
    #                             mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])

    # model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
    #                     args.samples_per_epoch,
    #                     args.nb_epoch,
    #                     verbose=1,
    #                     #callbacks=[checkpoint],
    #                     validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
    #                     validation_steps=len(X_valid),
    #                     max_queue_size=1)

    # print("THIS IS MY IMAGE AND STEERING")
    #images, steers = batch_generator(args.data_dir, X_train, y_train, True)
    # print(images[0])
    # print(images.shape)
    # print(len(images))
    # print(steers[0:1000])
    # print(len(steers))

    #val_img, val_steer = batch_generator(args.data_dir, X_valid, y_valid, False)
    # print(len(X_train)//args.batch_size)
    history = model.fit(batch_generator(args.data_dir, X_train, y_train, 1500, True), epochs=args.nb_epoch, validation_data=batch_generator(args.data_dir, X_valid, y_valid, 1500, False), batch_size=args.batch_size, verbose=1, shuffle=1, steps_per_epoch = args.samples_per_epoch)

    # predictions = model.evaluate(X_valid, y_valid)
    # print(predictions)
    model.save('model', save_format='h5')

    print(history.history)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.legend(['accuracy', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
def s2b(s):

    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def main():

    parser = argparse.ArgumentParser(description='Behaivioral Cloning Training Program')
    parser.add_argument('-d', help = 'data directory', dest= 'data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='tdrop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=25)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=58)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-3)
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
