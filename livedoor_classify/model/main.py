from os import environ
from pathlib import Path
import numpy as np
import requests
import json
import pickle
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim import matutils

DIC = "livedoordic.txt"
TFIDF_MODEL = "tfidf_model.model"
LSI_MODEL = "lsi_model.model"
SVC_MODEL = 'SVC_model.sav'
XGB_MODEL = 'Xgboost_model.sav'
LGBM_MODEL = 'LightGBM_model.sav'

USER_ID = Path(environ.get('USER_ID', '.'))
ACCESS_TOKEN = Path(environ.get('ACCESS_TOKEN', '.'))

# ここにmecabサーバのURL設定する
MECAB_URL = "XXXXX"

label = ['dokujo-tsushin',
 'it-life-hack',
 'kaden-channel',
 'livedoor-homme',
 'movie-enter',
 'peachy',
 'smax',
 'sports-watch',
 'topic-news']

def mecab(text):

    dic = {"text":text}
    text = json.dumps(dic)
    auth = (USER_ID,ACCESS_TOKEN)
    response = requests.post(MECAB_URL, text,headers={'Content-Type': 'application/json; charset=UTF-8'}, auth=auth)

    return response.json()

def process_text(json_file):

    text = json_file["text"]

    #test用のデータをMeCabにかける
    mecab_list = mecab(text)

    word_list = []
    doc_list = []

    #名詞だけ取得し、成形する処理
    text_data_list = mecab_list.split("\n")    
    for text in text_data_list:
        if text == 'EOS':
            break
        else:
            text = text.split("\t")
            word = text[0]
            word_meta = text[1]
                
            word_detail = word_meta.split(",")
            if word_detail[0] == '名詞':
                doc_list.append(word)
    doc_list = [doc_list]

    # 辞書のLOAD
    dic = Dictionary.load_from_text(DIC)
    bow_corpus = [dic.doc2bow(d) for d in doc_list]

    # tfidfモデルのLOAD
    tfidf_model  = TfidfModel.load(TFIDF_MODEL)
    tfidf_corpus = tfidf_model[bow_corpus]

    # LSIモデルのロード
    lsi_model = LsiModel.load(LSI_MODEL)
    lsi_corpus = lsi_model[tfidf_corpus]

    #gensimコーパスからdenseへ
    dime = 100
    test_dense = list(matutils.corpus2dense(lsi_corpus, num_terms=dime).T)

    #出力json
    json = {}

    #SVCモデルのロードと推論
    loaded_model = pickle.load(open(SVC_MODEL, 'rb'))
    pre = loaded_model.predict(test_dense)
    json["SVC"] = label[int(pre)]

    #XGBoostモデルのロードと推論
    loaded_model = pickle.load(open(XGB_MODEL, 'rb'))
    pre = result = loaded_model.predict(test_dense)
    json["XGB"] = label[int(pre)]

    #LightGBMモデルのロードと推論
    loaded_model = pickle.load(open(LGBM_MODEL, 'rb'))
    result_list = loaded_model.predict(test_dense)
    pre = np.argmax(result_list, axis=1)
    json["LGBM"] = label[int(pre)]

    return json


def handler(iter, context):
    for json_file in iter:
        yield process_text(json_file)
