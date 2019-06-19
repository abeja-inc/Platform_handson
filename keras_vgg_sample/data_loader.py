"""
the following labeling rule is used in the dataset.

 animal | label  | category
--------|--------|----------
   dog  |   1    |    0
   cat  |   2    |    1
"""
import io
import os
import re
from typing import Generator, Tuple
import zipfile

from abeja.datasets import Client
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image
import requests


def __img_to_processed_arr(img):
    img_arr = img_to_array(img)
    img_arr = preprocess_input(img_arr, mode='tf')
    return img_arr


def download(url, filename):
    res = requests.get(url, stream=True, allow_redirects=True)
    with open(filename, 'wb') as file:
        for chunk in res:
            file.write(chunk)


def load_dataset_from_api(
        dataset_id: str, img_rows: int = 128, img_cols: int = 128) \
        -> Generator[Tuple[np.ndarray, str], None, None]:
    client = Client()
    dataset = client.get_dataset(dataset_id)

    for item in dataset.dataset_items.list(prefetch=True):
        file_content = item.source_data[0].get_content()
        label_id = item.attributes['classification'][0]['label_id']
        category_id = str(int(label_id) - 1)

        file_like_object = io.BytesIO(file_content)
        img = load_img(file_like_object, target_size=(img_rows, img_cols))
        img_arr = __img_to_processed_arr(img)
        yield img_arr, category_id

        # augment process
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img_arr = __img_to_processed_arr(flipped_img)
        yield flipped_img_arr, category_id


def load_dataset_from_path(
        target_path: str, img_rows: int = 128, img_cols: int = 128) \
        -> Generator[Tuple[np.ndarray, str], None, None]:
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # download zip
    url = 'https://s3-ap-northeast-1.amazonaws.com/abeja-platform-code-samples-prod/data/cats_dogs.zip'
    zipfile_name = url.split('/')[-1]
    zipfile_path = f'{target_path}/{zipfile_name}'
    if not os.path.exists(zipfile_path):
        download(url, zipfile_path)

    # extract zip
    data_path = f'{target_path}/cats_dogs'
    if not os.path.exists(data_path):
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(target_path)

    for label in os.listdir(data_path):
        if not re.match(r'^\d+$', label):
            continue

        category_dir = os.path.join(data_path, label)
        for filename in os.listdir(category_dir):
            if not re.match(r'^\d+\..+$', filename):
                continue

            filepath = os.path.join(data_path, label, filename)
            category_id = str(int(label) - 1)

            img = load_img(filepath, target_size=(img_rows, img_cols))
            img_arr = __img_to_processed_arr(img)
            yield img_arr, category_id

            # augment process
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img_arr = __img_to_processed_arr(flipped_img)
            yield flipped_img_arr, category_id

