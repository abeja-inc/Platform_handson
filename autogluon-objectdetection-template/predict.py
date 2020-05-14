import http
import os
import io
from autogluon import Detector


ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')

def handler(request, context):

    contents = request['contents']
    content = contents[0].read()

    with open("input_file.jpg","wb") as f:
        f.write(content)
 
    weight_file_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pkl')
    detector = Detector.load(weight_file_path)   
    ind, prob, loc = detector.predict("input_file.jpg")

    result = {
        'boxes': [],
        'classes': [],
        'scores': []
    }

    box = []
    for data in loc:
        height_1 = data[0] * 0.75
        width_1 = data[1] * 0.75
        height_2 = data[2] * 0.75
        width_2 = data[3] *0.75
        box.append([height_1,width_1, height_2, width_2])
    result['boxes'] = box
    result['classes'] = ind
    result['scores']  = prob


    return {
        'status_code': http.HTTPStatus.OK,
        'content_type': 'application/json',
        'content': result
    }
