import sys
import MeCab

def mecab_result(dic_data):

    m = MeCab.Tagger("-d ./neologd")

    sentence = ""

    for x in dic_data:
        word = m.parse(dic_data[x])
        sentence = sentence + str(word)

    return sentence
    
def handler(iter, context):
    for json_file in iter:
        yield mecab_result(json_file)

