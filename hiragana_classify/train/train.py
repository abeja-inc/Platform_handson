from os import environ
from pathlib import Path
import subprocess

import numpy as np
from keras import layers, models, optimizers
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from abeja.contrib.keras.callbacks import Statistics

TRAINING_JOB_DEFINITION_NAME = environ['TRAINING_JOB_DEFINITION_NAME']
TRAINING_JOB_DEFINITION_VERSION = environ.get(
    'TRAINING_JOB_DEFINITION_VERSION', 0)
TRAINING_JOB_ID = environ.get('TRAINING_JOB_ID', 0)

EPOCHS = int(environ.get('EPOCHS', 20))
BATCH_SIZE = int(environ.get('BATCH_SIZE', 128))
IMAGE_SIZE = int(environ.get('IMAGE_SIZE', 48))
RANDOM_SEED = int(environ.get('RANDOM_SEED', 106201324))
USE_DATASET_CACHE = bool(int(environ.get('USE_DATASET_CACHE', 1)))


ABEJA_TRAINING_RESULT_DIR = Path(environ.get('ABEJA_TRAINING_RESULT_DIR', '.'))
ABEJA_STORAGE_DIR_PATH = Path(environ.get('ABEJA_STORAGE_DIR_PATH', '.'))


DATASET_CACHE_ROOT = ABEJA_STORAGE_DIR_PATH / \
    'cache' / TRAINING_JOB_DEFINITION_NAME / \
    str(TRAINING_JOB_DEFINITION_VERSION)
DATASET_EXTRACT_DIR = Path('/home/data')
DATASET_CACHE_FILE = DATASET_CACHE_ROOT / f'hiragana73.cache-{IMAGE_SIZE}.npz'


def load_dataset():
    try:
        if USE_DATASET_CACHE:
            with np.load(DATASET_CACHE_FILE) as data:
                return data['X'], data['Y']
    except IOError:
        print(f'No cached dataset found at {DATASET_CACHE_FILE}')

    # download
    zip_file = DATASET_CACHE_ROOT / 'hiragana73.zip'
    dist_path = DATASET_EXTRACT_DIR / 'hiragana73'

    if not zip_file.exists():
        subprocess.run(['wget', 'http://lab.ndl.go.jp/dataset/hiragana73.zip',
                        '-P', DATASET_CACHE_ROOT], check=True)

    if not dist_path.exists():
        subprocess.run(['unzip', '-o', '-q',
                        zip_file,
                        '-d', DATASET_EXTRACT_DIR], check=True)

    # Preprocessing
    folder = list(dist_path.iterdir())

    # -- Sort paths to convert its name to index
    folder.sort()

    X = []
    Y = []
    for index, name in tqdm(enumerate(folder), total=len(folder)):

        # フォルダの一覧をリストとして取得する
        dir = dist_path / name
        files = dir.glob("*.png")

        for i, file in enumerate(files):
            image = Image.open(file)
            image = image.convert("RGB")
            # image = image.convert('L')
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            data = np.asarray(image)
            X.append(data)
            Y.append(index)

    X, Y = np.array(X), np.array(Y)

    if USE_DATASET_CACHE:
        DATASET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        np.savez(DATASET_CACHE_FILE, X=X, Y=Y)

    return X, Y


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(73, activation='softmax'))

    return model


def handler(context):
    print('Preprocessing images...')

    X, Y = load_dataset()

    # 学習とテストで 8:2 のデータ割合に分ける
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, test_size=0.2, random_state=RANDOM_SEED)

    print(f'X_train.shape = {X_train.shape}')
    print(f'Y_train.shape = {Y_train.shape}')
    print(f'X_test.shape = {X_test.shape}')
    print(f'Y_test.shape = {Y_test.shape}')

    # 0-255の整数値を0〜1の小数に変換する(正規化)
    X_train_orig = X_train
    X_test_orig = X_test
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # one-hot vector形式に変換する
    Y_train = np_utils.to_categorical(Y_train, 73)
    Y_test = np_utils.to_categorical(Y_test, 73)

    log_path = ABEJA_TRAINING_RESULT_DIR / 'logs'
    tensorboard = TensorBoard(log_dir=str(log_path), histogram_freq=0,
                              write_graph=True, write_images=False)

    model = build_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=[tensorboard, Statistics()])

    # モデルの評価と保存
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(str(ABEJA_TRAINING_RESULT_DIR / 'hiragana_model.h5'),
               include_optimizer=False)


if __name__ == '__main__':
    handler({})
