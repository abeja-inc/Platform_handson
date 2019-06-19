import os
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image

img_rows, img_cols = 128, 128
model = load_model(os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'model.h5'))


def decode_predictions(result):
    categories = {
        0: 'dog',
        1: 'cat'
    }
    result_with_labels = [{"label": categories[i], "probability": score} for i, score in enumerate(result)]
    return sorted(result_with_labels, key=lambda x: x['probability'], reverse=True)


def handler(_iter, ctx):
    for img in _iter:
        img = Image.fromarray(img)
        img = img.resize((img_rows, img_cols))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='tf')

        result = model.predict(x)[0]
        sorted_result = decode_predictions(result.tolist())
        yield {"result": sorted_result}

