from collections import defaultdict
import nltk
import pymorphy2
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from math import log
import json
import scipy
import numpy as np

morph = pymorphy2.MorphAnalyzer()
main_dir = 'D:\\work\\ir\\corp'
save_dir = 'D:\\work\\ir\\db_okapi'
K1 = 2.0
B = 0.75


def fn_tdm_df(docs, docs_names=None):
    vectorizer = CountVectorizer()
    x1 = vectorizer.fit_transform(docs)
    df = pd.DataFrame(x1.toarray().transpose(),
                      index=vectorizer.get_feature_names())
    if docs_names is not None:
        df.columns = docs_names
    return df


def create_db():
    files_list = []
    filenames_list = []
    for root, dirs, filenames in os.walk(main_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            filenames_list.append(os.path.splitext(filename)[0].split('-')[-1].strip())
            with open(filepath, 'r',encoding='utf-8') as friends_txt:
                files_list.append(friends_txt.read())

    files_list = [text.split('text:')[-1] for text in files_list]
    tokens_list = []
    for file in files_list:
        parsed_line = []
        tokens = nltk.word_tokenize(file)
        for token in tokens:
            new_token = morph.parse(token)[0].normal_form
            parsed_line.append(new_token)
        parsed_line = " ".join(parsed_line)
        tokens_list.append(parsed_line)
    tdm = fn_tdm_df(tokens_list, docs_names = filenames_list)
    filename_lengths_src = []
    for tokens in tokens_list:
        filename_lengths_src.append(len(tokens.split()))
    length_series = pd.Series(filename_lengths_src)
    length_series.index = filenames_list
    with open(os.path.join(save_dir, 'fnames.json'), 'w' ,encoding='utf-8') as fnames_file:
        fnames_file.write(json.dumps(filenames_list, ensure_ascii=False))
    length_series.to_csv(os.path.join(save_dir,'lenser.csv'))
    inverted_idx = tdm.transpose()
    sparse_df = scipy.sparse.csr_matrix(inverted_idx.values)
    pd.Series(inverted_idx.index).to_csv(os.path.join(save_dir,'iidx_idx'), index=False)
    pd.Series(inverted_idx.columns).to_csv(os.path.join(save_dir,'iidx_cols'), index=False)
    scipy.sparse.save_npz((os.path.join(save_dir,'iidx')), sparse_df)


def okapi_load():
    with open(os.path.join(save_dir, 'fnames.json'), 'r', encoding='utf-8') as json_fnames:
        fnames = json.load(json_fnames)
    sparse = scipy.sparse.load_npz(os.path.join(save_dir,'iidx.npz'))
    index = pd.read_csv(os.path.join(save_dir, 'iidx_idx'), header=None)
    columns = pd.read_csv(os.path.join(save_dir, 'iidx_cols'), header=None)
    index, columns = [index[0] for index in index.values.tolist()], [column[0]for column in columns.values.tolist()]
    df = pd.DataFrame(sparse.todense(), index=index, columns=columns)
    return df, pd.read_csv(os.path.join(save_dir, 'lenser.csv',),  header=None, index_col=0), fnames


def score_BM25(qf, dl, avgdl,  N, n, k1=K1, b=B):
    """
    Compute similarity score between word and document from collection
    :return: score
    """
    idf = log((N -n +0.5)/(n + 0.5))
    numerator = (k1 + 1) * qf
    denumerator = qf + k1* (1 - b + b  * (dl / avgdl))
    return float(idf * numerator / denumerator)


def compute_sim(lemma, inverted_index, relevance_dict, length_series) -> float:
    """
    Compute similarity score between word in search query and all document  from collection
    :return: score
    """
    doc_freqs = inverted_index[lemma]
    n = doc_freqs.astype('bool').sum()
    N = len(doc_freqs.index)
    avgdl = length_series.mean()

    for doc in doc_freqs.index:
        qf = doc_freqs.loc[doc]
        dl = length_series.loc[doc]
        relevance_dict[doc] += score_BM25(qf, dl, avgdl, N, n)
    return relevance_dict
inverted_idx, length_series, filenames_list = okapi_load()

def okapi_search(query) -> list:
    """
    Compute sim score between search query and all documents in collection
    Collect as pair (doc_id, score)
    :param query: input text
    :return: list of lists with (doc_id, score)
    """
    relevance_dict = defaultdict(float)
    for word in query.split():
        relevance_dict = compute_sim(morph.parse(word)[0].normal_form, inverted_idx, relevance_dict, length_series)
    relevance_series = pd.Series(list(relevance_dict.values()))

    relevance_series.index = filenames_list
    sorted_relevance = relevance_series.sort_values(ascending=False)
    return sorted_relevance


