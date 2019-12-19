# prediction
import os
import json
import http
from sklearn.externals import joblib

clf = joblib.load(os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'), 'model.pkl'))
print(clf)


def decode_predictions(result):
    categories = {
        0: 'False',
        1: 'True',
    }
    result_with_labels = [{"toxic": categories[i], "probability": score} for i, score in enumerate(result)]
    return sorted(result_with_labels, key=lambda x: x['probability'], reverse=True)


def handler(request, context):
    req_data = json.load(request)
    print(req_data)
    comment = req_data['comment']
    print(comment)
    result = clf.predict_proba([comment])[0]
    print(result)
    sorted_result = decode_predictions(result.tolist())

    return {
        'status_code': http.HTTPStatus.OK,
        'content_type': 'application/json; charset=utf8',
        'content': {"result": sorted_result}
        }
