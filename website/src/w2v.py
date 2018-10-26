from gensim.models import Word2Vec, KeyedVectors
import nltk
import pymorphy2
import os
from collections import defaultdict
import pandas as pd
from russian_tagsets import converters
from gensim import matutils
import numpy as np
import json



model_path = 'D:\\work\\ir\\ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec'
# model = KeyedVectors.load_word2vec_format(model_path, binary=False)
to_ud = converters.converter('opencorpora-int', 'ud20')
morph = pymorphy2.MorphAnalyzer()
main_dir = 'D:\\work\\ir\\corp'
save_dir = 'D:\\work\\ir\\db_w2v\\'


def get_w2v_vectors(tokens_string):
    """Получает вектор документа"""
    vec = np.zeros((300))
    counter = 0
    for token in tokens_string.split():
        if token in model:
            counter += 1
            vec = np.add(vec, model[token])
    if counter > 0:
        return vec/counter
    else:
        return vec

def create_db():
    files_list = []
    filenames_list = []

    for root, dirs, filenames in os.walk(main_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            filenames_list.append(os.path.splitext(filename)[0].split('-')[-1].strip())
            with open(filepath, 'r',encoding='utf-8') as txt_file:
                files_list.append(txt_file.read())

    files_list = [text.split('text:')[-1] for text in files_list]

    tokens_list = []

    for file in files_list:
        parsed_line = []
        tokens = nltk.word_tokenize(file)
        for token in tokens:
            res = morph.parse(token)[0]
            if res.tag.POS:
                new_token =  res.normal_form +'_'+ to_ud(res.tag.POS).split()[0]
                parsed_line.append(new_token)
        parsed_line = " ".join(parsed_line)
        tokens_list.append(parsed_line)
    for idx, token_string in enumerate(tokens_list):
        np.save( save_dir + filenames_list[idx] + '.npy', get_w2v_vectors(token_string))
    with open(os.path.join(save_dir, 'fnames.json'), 'w' ,encoding='utf-8') as fnames_file:
        fnames_file.write(json.dumps(filenames_list, ensure_ascii=False))



def load_db():
    with open(os.path.join(save_dir, 'fnames.json'), 'r', encoding='utf-8') as json_fnames:
        fnames = json.load(json_fnames)
    return fnames



def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)


def search_w2v(query, model):
    filenames_list = load_db()
    relevance_dict = defaultdict(float)
    words = query.split()
    for word in words:
        res = morph.parse(word)[0]
        if res.tag.POS:
            new_token =  res.normal_form +'_'+ to_ud(res.tag.POS).split()[0]
            relevance_dict = w2v_lemma_sim(new_token, relevance_dict, filenames_list, model)
        print(res.tag.POS)
    relevance_series = pd.Series(list(relevance_dict.values()))
    print(relevance_series)
    relevance_series.index = filenames_list
    sorted_relevance = relevance_series.sort_values(ascending=False)
    return sorted_relevance

def w2v_lemma_sim(lemma, relevance_dict, filenames_list, model):
    for filename in filenames_list:
        doc_vec = np.load(save_dir + filename + '.npy')
        if lemma in model:
            lemma_vec = model[lemma]
        else:
            lemma_vec = np.zeros(300)
        relevance_dict[filename] += similarity(doc_vec, lemma_vec)
    return relevance_dict


