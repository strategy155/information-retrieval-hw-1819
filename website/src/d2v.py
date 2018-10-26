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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument




model_path = 'D:\\work\\ir\\d2vec_avito.vec'
to_ud = converters.converter('opencorpora-int', 'ud20')
morph = pymorphy2.MorphAnalyzer()
main_dir = 'D:\\work\\ir\\corp'
save_dir = 'D:\\work\\ir\\db_d2v\\'

model = Doc2Vec.load(model_path)

def get_d2v_vectors(d2v, filename):
    """Получает вектор документа"""
    return d2v.docvecs[filename]


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

    docs = []

    for idx, file in enumerate(files_list):
        parsed_line = []
        tokens = nltk.word_tokenize(file)
        for token in tokens:
            res = morph.parse(token)[0]
            if res.tag.POS:
                new_token = res.normal_form + '_' + to_ud(res.tag.POS).split()[
                    0]
                parsed_line.append(new_token)
        t = TaggedDocument(parsed_line, [filenames_list[idx]])
        docs.append(t)
    tagged_data = docs
    d2model = Doc2Vec(vector_size=100, min_count=1, alpha=0.025,
                min_alpha=0.025, epochs=100, workers=4, dm=1)
    d2model.build_vocab(tagged_data)
    d2model.train(tagged_data, total_examples=d2model.corpus_count, epochs=d2model.epochs)
    for filename  in filenames_list:
        np.save( save_dir + filename + '.npy', get_d2v_vectors(d2model, filename))
    d2model.save(model_path)
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


def d2v_lemma_sim(lemma, relevance_dict, filenames_list):
    for filename in filenames_list:
        doc_vec = np.load(save_dir + filename + '.npy')
        if lemma in model.wv:
            lemma_vec = model.wv[lemma]
        else:
            lemma_vec = np.zeros(100)
        relevance_dict[filename] += similarity(doc_vec, lemma_vec)
    return relevance_dict


def search_d2v(query):
    filenames_list = load_db()
    relevance_dict = defaultdict(float)
    words = query.split()
    for word in words:
        res = morph.parse(word)[0]
        if res.tag.POS:
            new_token = res.normal_form + '_' + to_ud(res.tag.POS).split()[0]
            relevance_dict = d2v_lemma_sim(new_token, relevance_dict, filenames_list)
    relevance_series = pd.Series(list(relevance_dict.values()))
    relevance_series.index = filenames_list
    sorted_relevance = relevance_series.sort_values(ascending=False)
    return sorted_relevance


