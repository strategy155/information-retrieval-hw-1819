from flask import Flask, url_for, redirect, abort
from flask import request
from flask import render_template
from flask_wtf import Form
from wtforms import validators, RadioField
from src.okapi import okapi_search
from src.w2v import search_w2v
from src.d2v import search_d2v
from functools import partial

from gensim.models import Word2Vec, KeyedVectors



app = Flask(__name__)

app.config['TESTING'] = True

app.secret_key = 'TEST'

model_path = 'D:\\work\\ir\\ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec'
model = KeyedVectors.load_word2vec_format(model_path, binary=False)


corp_path = 'D:\\work\\ir\\corp\\'

def search(query, type):
    if type == 'inv':
        search_func = okapi_search
    elif type == 'w2v':
        search_func = partial(search_w2v, model=model)
    else:
        search_func = search_d2v
    try:
        best_commercials = (search_func(query).index[:30])
    except ValueError:
        abort(406)
    best_commercials_response = [(url_for('commercial', commercial_name=name), name) for name in best_commercials]
    return best_commercials_response

class SearchForm(Form):
    choice_switcher = RadioField(
        'Choice?',
        [validators.DataRequired()],
        choices=[('inv', 'Inverted + Okapi'), ('w2v', 'Word2Vec'), ('d2v', 'Doc2Vec')]
    )



@app.errorhandler(406)
def page_not_found(e):
    return render_template('errorpage.html'), 406

@app.route('/',  methods=['GET', 'POST'])
def index():
    search_form = SearchForm()
    print(request.args)
    if request.args:
        query = request.args['query']
        links = search(query, request.args['choice_switcher'])
        return render_template('index.html', links=links, form=search_form )
    return render_template('index.html',links=[], form=search_form)

@app.route('/commercials/<commercial_name>',  methods=['GET'])
def commercial(commercial_name):
    with open(corp_path + commercial_name + '.txt', 'r', encoding='utf-8') as comm_file:
        commercial_text = comm_file.read()
        commercial_text = commercial_text.replace(':', ':\n')
    return render_template('commercial.html',name = commercial_name, text = commercial_text)



if __name__ == '__main__':
    app.run(host='localhost', port=5005, debug=True)

    

