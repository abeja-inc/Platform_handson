#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import json
from pathlib import Path

import lightgbm as lgb
from tensorboardX import SummaryWriter

from callbacks import Statistics, TensorBoardCallback, ModelExtractionCallback
from data_loader import train_data_loader
from parameters import Parameters


ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')
Path(ABEJA_TRAINING_RESULT_DIR).mkdir(exist_ok=True)

DATALAKE_CHANNEL_ID = Parameters.DATALAKE_CHANNEL_ID
DATALAKE_TRAIN_FILE_ID = Parameters.DATALAKE_TRAIN_FILE_ID
DATALAKE_VAL_FILE_ID = Parameters.DATALAKE_VAL_FILE_ID
INPUT_FIELDS = Parameters.INPUT_FIELDS
LABEL_FIELD = Parameters.LABEL_FIELD
PARAMS = Parameters.as_params()

STRATIFIED = Parameters.STRATIFIED and Parameters.IS_CLASSIFICATION
IS_MULTI = Parameters.OBJECTIVE.startswith("multi")

statistics = Statistics(Parameters.NUM_ITERATIONS)

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)


# In[4]:


print(f'start training with parameters : {Parameters.as_dict()}')


# In[5]:


X_train, y_train, cols_train = train_data_loader(
    DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_FILE_ID, LABEL_FIELD, INPUT_FIELDS)


# In[8]:


dtrain = lgb.Dataset(X_train, y_train)

if DATALAKE_VAL_FILE_ID:
    X_val, y_val, _ = train_data_loader(
        DATALAKE_CHANNEL_ID, DATALAKE_VAL_FILE_ID, LABEL_FIELD, INPUT_FIELDS)
else:
    X_val, y_val = None, None

extraction_cb = ModelExtractionCallback()
tensorboard_cb = TensorBoardCallback(statistics, writer)
tensorboard_cb.set_valid(X_val, y_val, Parameters.IS_CLASSIFICATION, IS_MULTI, Parameters.NUM_CLASS)
callbacks = [extraction_cb, tensorboard_cb,]


# In[9]:


lgb.cv(PARAMS, dtrain, nfold=Parameters.NFOLD,
       early_stopping_rounds=Parameters.EARLY_STOPPING_ROUNDS,
       verbose_eval=Parameters.VERBOSE_EVAL,
       stratified=STRATIFIED,
       callbacks=callbacks,
       metrics=Parameters.METRIC,
       seed=Parameters.SEED)


# In[10]:


models = extraction_cb.raw_boosters
for i,model in enumerate(models):
    model.save_model(os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.txt'))

di = {
    **(Parameters.as_dict()),
    'cols_train': cols_train
}
lgb_env = open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'lgb_env.json'), 'w')
json.dump(di, lgb_env)
lgb_env.close()
writer.close()


# In[ ]:


def handler(context):
    print("finish.")


# In[ ]:


parameters = Parameters.as_env()
parameters.update({'IS_CLASSIFICATION': 'False'})


# In[ ]:




