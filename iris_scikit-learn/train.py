# import library
import numpy as np
import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.externals import joblib

from abeja.datalake import Client as DatalakeClient
from abeja.train import Client as TrainClient
from abeja.train.statistics import Statistics as ABEJAStatistics

# define training result dir
ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')

# define parameters
epochs = int(os.environ.get('NUM_EPOCHS', 1))
c = float(os.environ.get('C', 1))

# define datalake channel_id and file_id
channel_id = os.environ.get('CHANNEL_ID', 'XXXXXXXXXX')


def load_latest_file_from_datalake(channel_id):
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(channel_id)

    # list all file objects
    files = [f for f in channel.list_files()]

    # sort file objects by uploaded datetime
    files_sorted_by_uploaded_time = sorted(files, key=lambda files: files.uploaded_at)

    # select the latest file object
    latest_file = files_sorted_by_uploaded_time[-1]

    # get a file path
    latest_file_path = latest_file.download_url

    # print the uploaded datetime of the selected file object
    latest_file_path_datetime = latest_file.uploaded_at
    print('load file uploaded at {} (UTC time).'.format(latest_file_path_datetime))

    return latest_file_path


def handler(context):
    """
    the following csv file should be stored in the datalake channel.

     sepal_lenght (cm)| sepal_width (cm)| petal_lenght (cm)| petal_width (cm)|  target  |
    ------------------|-----------------|------------------|-----------------|----------|
         float        |     float       |     float        |     float       |    int   |

    """
    iris = datasets.load_iris()
    file_path = load_latest_file_from_datalake(channel_id)
    data = pd.read_csv(file_path, sep=',')
    X = data[iris.feature_names].values.astype('float64')
    Y = data['target'].values.astype('int64')
    print('successfully load datalake channel file.')

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)

    # define model
    model = LogisticRegression(solver='lbfgs', C=c,
                               multi_class='multinomial', max_iter=epochs)

    # train model
    model.fit(X_train, Y_train)

    # evaluate model
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    train_loss = log_loss(Y_train, model.predict_proba(X_train))
    valid_acc = accuracy_score(Y_test, model.predict(X_test))
    valid_loss = log_loss(Y_test, model.predict_proba(X_test))

    # update ABEJA statisctics
    train_client = TrainClient()
    statistics = ABEJAStatistics(num_epochs=epochs, epoch=epochs)
    statistics.add_stage(name=ABEJAStatistics.STAGE_TRAIN,
                         accuracy=train_acc,
                         loss=train_loss)
    statistics.add_stage(name=ABEJAStatistics.STAGE_VALIDATION,
                         accuracy=valid_acc,
                         loss=valid_loss)
    train_client.update_statistics(statistics)
    print('Train accuracy is {:.3f}.'.format(train_acc))
    print('Train loss is {:.3f}.'.format(train_loss))
    print('Valid accuracy is {:.3f}.'.format(valid_acc))
    print('Valid loss is {:.3f}.'.format(valid_loss))

    # save model
    joblib.dump(model, os.path.join(ABEJA_TRAINING_RESULT_DIR, 'model.pkl'))
