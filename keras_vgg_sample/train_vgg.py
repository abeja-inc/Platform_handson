"""
cats and dogs classifier
"""
import os
import random

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras import backend as K
from keras import applications
from keras.callbacks import TensorBoard, EarlyStopping
from abeja.contrib.keras.callbacks import Statistics

from data_loader import load_dataset_from_api
from data_loader import load_dataset_from_path

num_classes = 2
batch_size = int(os.environ.get('BATCH_SIZE', 32))
epochs = int(os.environ.get('NUM_EPOCHS', 5))

# input image dimensions
img_rows, img_cols = 128, 128
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
ARTIFACT_FILE_NAME = os.environ.get('ARTIFACT_FILE_NAME', 'model.h5')

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')


def create_model(input_shape):
    # instantiate vgg without fully-connected layer.
    vgg16 = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=input_shape)

    # create fully-connected layer
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))

    # create new model
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # fix vgg layers
    for layer in model.layers[:15]:
        layer.trainable = False

    return model


def handler(context):
    dataset_alias = context.datasets
    if dataset_alias.get('train'):
        print('load dataset from api')
        dataset_id = dataset_alias['train']  # set alias specified in console
        data = list(load_dataset_from_api(dataset_id))
    else:
        print('load dataset from path')
        dataset_path = '.'
        if os.path.exists('/mnt'):
            dataset_path = '/mnt/data'
        data = list(load_dataset_from_path(dataset_path))
    random.shuffle(data)

    x = np.array([_[0] for _ in data])
    y = np.array([_[1] for _ in data])
    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    nb_channels = 3
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], nb_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], nb_channels, img_rows, img_cols)
        input_shape = (nb_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, nb_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, nb_channels)
        input_shape = (img_rows, img_cols, nb_channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = create_model(input_shape)

    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                              write_graph=True, write_images=False)
    statistics = Statistics()
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    # Do you want to add `checkpoint` to callback as well?
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # fit_generator
    # image loader
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard, statistics, early])
    score = model.evaluate(x_test, y_test, verbose=0)
    model.save(os.path.join(ABEJA_TRAINING_RESULT_DIR, ARTIFACT_FILE_NAME))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    from collections import namedtuple

    Context = namedtuple('Context', ('datasets',))
    handler(Context(datasets={}))

