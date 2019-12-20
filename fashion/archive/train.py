from __future__ import print_function
import os
import io
import random

import numpy as np
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import TensorBoard
from abeja.datasets import Client

from utils.callbacks import Statistics


batch_size = 32
num_classes = 5
epochs = int(os.environ.get('NUM_EPOCHS', '100'))
DROPOUT = float(os.environ.get('DROPOUT', '1.0'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '42'))
HORIZONTAL_FLIP = bool(os.environ.get(
    'HORIZONTAL_FLIP', 'False').lower() == 'true')
random.seed(RANDOM_SEED)

# input image dimensions
img_rows, img_cols = 224, 224
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')


def load_dataset_from_api(dataset_id):
    """
    the following labeling rule is used in the dataset.

     label          | label_id
    ----------------|--------
       compressor   |   0
       fan_motor    |   1
       filter       |   2
       louver_motor |   3
       thermistor   |   4
    """
    client = Client()
    dataset = client.get_dataset(dataset_id)

    for item in dataset.dataset_items.list(prefetch=True):
        file_content = item.source_data[0].get_content()
        label = item.attributes['classification'][0]['label_id']
        file_like_object = io.BytesIO(file_content)
        img = load_img(file_like_object, target_size=(img_rows, img_cols))
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

        x = img_to_array(img)
        y = img_to_array(flipped_img)
        x = preprocess_input(x, mode='tf')
        y = preprocess_input(y, mode='tf')
        yield x, label
        if HORIZONTAL_FLIP:
            yield y, label

    raise StopIteration


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROPOUT))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def handler(context):
    dataset_alias = context.datasets
    dataset_id = dataset_alias['data']   # set alias specified in console
    data = list(load_dataset_from_api(dataset_id))
    # data = list(load_dataset_from_path())
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
    # Do you want to add `checkpoint` to callback as well?
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['accuracy'])

    # fit_generator
    # image loader
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard, statistics])
    score = model.evaluate(x_test, y_test, verbose=0)
    model.save(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.h5'))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    handler(None, None)
