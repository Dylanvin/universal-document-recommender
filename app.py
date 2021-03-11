from flask import Flask, render_template, request
from forms import RunSystemForm
from wtforms.validators import ValidationError
from algorithms.tfidf_document_similarity import TfIdf
from algorithms.doc2vec_document_similarity import D2V
from algorithms.lsa_document_similarity import LSA
from algorithms.bert_document_similarity import BERT
from algorithms.evaluate import Evaluate
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
import os.path
import pickle
import numpy as np
import nltk
import html2text
import requests
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


@app.route('/text/<doc>')
def text(doc):
    doc = doc.replace('FORWARD_SLASH', '/')
    return render_template('text.html', doc=doc)


@app.route('/', methods=['post', 'get'])
def index():
    form = RunSystemForm()
    form.category.choices = list(target_names)
    if request.method == 'POST':

        check = form.showTextBox.data
        # getting webpage args
        if form.validate_on_submit():
            print(form.algorithm.data)
            print(form.query.data)
            print(form.category.data)


            if check:
                query_doc = request.form['query']
            else:

                URL = request.form['query_url']  ######implement URL form element option
                page = requests.get(URL).text
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                query_doc = h.handle(page)
                # except:
                #     return render_template('index.html', form=form)

            alg = request.form['algorithm']
            dist_method = request.form['measurement']
            category = request.form['category']
            n = int(request.form['num'])

            a = lambda x: " ".join(lemma.lemmatize(re.sub(r'[^a-zA-Z]', ' ', w).lower()) for w in x.split() if
                                   re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l and len(
                                       re.sub(r'[^a-zA-Z]', ' ', w)) > 2)

            filtered_query_doc = a(query_doc)

            if alg == "TfIdf":
                print("########################## TFIDF ##########################")
                ds = TfIdf()
                tf_idf = ds.tdfIdf(df, colnames, filtered_query_doc, category)
                docs = ds.similarDocs(tf_idf, len(df.index), dist_method,
                                      n)  # assumes query is the last doc in every value of key

            elif alg == "LSA":
                # LSA
                print("########################## LSA ##########################")
                ds = LSA()
                lsa = ds.tfidf_svd(df, colnames, filtered_query_doc, category)
                docs = ds.similar_docs(lsa, len(df.index), dist_method, n)


            elif alg == "Doc2Vec":
                # word2vec
                print("########################## WORD2VEC ##########################")
                ds = D2V()
                model = ds.train(df, colnames, model_file)
                docs = ds.similar_docs(model, doc2vec_vecs, filtered_query_doc, dist_method, n)
                # doc_list = []
                # dist_list = []
                # for i in docs:
                #     doc_list.append(i[0])
                #     dist_list.append(round(i[1], 5))


            elif alg == "BERT":
                # word2vec
                print("########################## BERT ##########################")
                ds = BERT()
                docs = ds.similar_docs(bert_vecs, filtered_query_doc, dist_method, n)

            doc_list = []
            dist_list = []
            for key, value in docs.items():
                doc_list.append(key)
                dist_list.append(round(value, 5))

            doc_d = df.iloc[doc_list]
            doc_d["Distance"] = dist_list
            eval = Evaluate()
            doc_cats = doc_d[colnames[0]].to_list()
            score = eval.get_score(doc_cats, category)
            doc_d = doc_d.to_dict('records')

            return render_template('results.html', doc_d=doc_d, doc_list=doc_list, category=category, score=score,
                                   dist_method=dist_method, alg=alg)

    return render_template('index.html', form=form)


def get_vecs(vec_file, alg):
    if not os.path.isfile(vec_file):
        if alg == "doc2vec":
            ds = D2V()
            model = ds.train(df, colnames, model_file)
            vecs = ds.create_vecs(model, df, colnames)
        elif alg == "bert":
            ds = BERT()
            vecs = ds.create_vecs(df, colnames)
        with open(vec_file, "wb") as f:
            pickle.dump(vecs, f)
    else:
        with open(vec_file, "rb") as f:
            vecs = pickle.load(f)
        vecs = np.array(vecs, float)

    return vecs


# pre-processing started
pd.set_option('display.max_colwidth', 200)
colnames = ['Category', 'Text']
# use sklearn 20newsgroups data set and use NLTK prepossessing methods
dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
target_names = dataset.target_names
df = pd.DataFrame({'Category': dataset.target, 'Text': dataset.data})
df = df.reset_index(drop=True)

for index, row in df.iterrows():
    df.loc[index, "Category"] = target_names[df.loc[index, "Category"]]
stop_words_l = stopwords.words('english')
lemma = WordNetLemmatizer()
df['Text cleaned'] = df.Text.apply(  # Cleaning
    lambda x: " ".join(lemma.lemmatize(re.sub(r'[^a-zA-Z]', ' ', w).lower()) for w in x.split() if
                       re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l and len(
                           re.sub(r'[^a-zA-Z]', ' ', w)) > 2))
df['Text'] = df['Text'].str.replace('/', 'FORWARD_SLASH')  # to prevent browser from getting confused

colnames.append('Text cleaned')
algorithms = ['tfidf', 'lsa', 'Doc2Vec']  # list of available methods to use
print("pre-processing done")

# doc2vec and BERT creating vecs
alg = "doc2vec"
model_file = 'd2v.model'
vec_file = 'doc2vec_vecs.txt'
doc2vec_vecs = get_vecs(vec_file, alg)

alg = "bert"
vec_file = 'bert_vecs.txt'
bert_vecs = get_vecs(vec_file, alg)
# pre-processing finished

print(__name__)
if __name__ == '__main__':
    print("test")
    app.run(host='0.0.0.0', threaded=True)
