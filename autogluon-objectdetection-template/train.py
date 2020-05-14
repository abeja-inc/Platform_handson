import os
from pathlib import Path
import autogluon as ag
from autogluon import ObjectDetection as task

root = './'
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')
EPOCHS = os.environ.get('EPOCHS','.')

filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',path=root)
filename = ag.unzip(filename_zip, root=root)

data_root = os.path.join(root, filename)
dataset_train = task.Dataset(data_root, classes=('motorbike',))
dataset_test = task.Dataset(data_root, index_file_name='test', classes=('motorbike',))

time_limits = 5*60*60  # 5 hours
epochs = int(EPOCHS)
detector = task.fit(dataset_train,num_trials=1,batch_size=16,epochs=epochs,lr=ag.Categorical(5e-4, 1e-4),ngpus_per_trial=1,time_limits=time_limits)

test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][1]))

savefile = Path(ABEJA_TRAINING_RESULT_DIR) / 'model.pkl'
detector.save(savefile)


