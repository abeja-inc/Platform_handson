# prediction
import os
import pandas as pd
import json
import numpy as np
from sklearn.externals import joblib

model = joblib.load(os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'model.pkl'))


def decode_predictions(result):
    categories = {
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    }
    result_with_labels = [{"label": categories[i], "probability": score} for i, score in enumerate(result)]
    return sorted(result_with_labels, key=lambda x: x['probability'], reverse=True)


def handler(_iter, ctx):
    '''
    _iter: json file
     {"iris": {"sepal_length (cm)": "XX",
               "sepal_width (cm)": "XX",
               "petal_length (cm)": "XX",
               "petal_width (cm)": "XX"}}
    '''
    for iter in _iter:
        sepal_length =iter['iris']['sepal_length (cm)']
        sepal_width = iter['iris']['sepal_width (cm)']
        petal_length = iter['iris']['petal_length (cm)']
        petal_width = iter['iris']['petal_width (cm)']
        x = np.array([[sepal_length, sepal_width, petal_length, petal_width]]).astype('float64')
        result = model.predict_proba(x)[0]
        sorted_result = decode_predictions(result.tolist())
        yield {"result": sorted_result}
