from os import environ
from pathlib import Path
from keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

IMAGE_SIZE = int(environ.get('IMAGE_SIZE', 48))
ABEJA_TRAINING_RESULT_DIR = Path(environ.get('ABEJA_TRAINING_RESULT_DIR', '.'))
MODEL_H5 = 'hiragana_model.h5'

if (ABEJA_TRAINING_RESULT_DIR / MODEL_H5).exists():
    model = load_model(str(ABEJA_TRAINING_RESULT_DIR / MODEL_H5))
else:
    model = load_model(str(Path(".") / MODEL_H5))


def process_image(img):
    img = Image.fromarray(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    label = [
        "あ", "い", "う", "え", "お",
        "か", "が", "き", "ぎ", "く", "ぐ", "け", "げ", "こ", "ご",
        "さ", "ざ", "し", "じ", "す", "ず", "せ", "ぜ", "そ", "ぞ",
        "た", "だ", "ち", "ぢ", "つ", "づ", "て", "で", "と", "ど",
        "な", "に", "ぬ", "ね", "の",
        "は", "ば", "ぱ", "ひ", "び", "ぴ", "ふ", "ぶ", "ぷ", "へ", "べ", "ぺ", "ほ", "ぼ", "ぽ",
        "ま", "み", "む", "め", "も",
        "や", "ゆ", "よ",
        "ら", "り", "る", "れ", "ろ",
        "わ", "ゐ", "ゑ", "を", "ん"]

    pred = model.predict(x, verbose=0)[0]
    result_with_labels = [{"label": label[i], "probability": score}
                          for i, score in enumerate(pred)]
    result = {"result": sorted(
        result_with_labels, key=lambda x: x['probability'], reverse=True)[:5]}

    return result


def handler(iter, context):
    for img in iter:
        yield process_image(img)
