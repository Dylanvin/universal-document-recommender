from flask import Flask, render_template, request
from forms import RunSystemForm
from algorithms.tfidf_document_similarity import TfIdf
from algorithms.doc2vec_document_similarity import D2V
from algorithms.lsa_document_similarity import LSA
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
    if form.validate_on_submit():
        print(form.algorithm.data)
        print(form.query.data)
        print(form.category.data)
    else:
        print(form.errors)

    if request.method == 'POST':
        # getting webpage args
        alg = request.form['algorithm']
        dist_method = request.form['measurement']
        query_doc = request.form['query']
        category = request.form['category']
        n = int(request.form['num'])

        # Cleaning query text
        query_doc_tokenized = word_tokenize(query_doc)
        query_doc_tokenized = [token.lower() for token in query_doc_tokenized]
        query_doc_tokenized = [word for word in query_doc_tokenized if not word in stopwords.words()]
        filtered_query_doc = (" ").join(query_doc_tokenized)
        print(filtered_query_doc)
        if alg == "TfIdf":
            print("########################## TFIDF ##########################")
            ds = TfIdf()
            tf_idf = ds.tdfIdf(df, colnames, filtered_query_doc, category)
            docs = ds.similarDocs(tf_idf, len(df.index), dist_method,
                                  n)  # assumes query is the last doc in every value of key
            doc_list = []
            dist_list = []
            for key, value in docs.items():
                doc_list.append(key)
                dist_list.append(round(value, 5))

        elif alg == "lsa":
            # LSA
            print("########################## LSA ##########################")
            ds = LSA()
            lsa = ds.tfidf_svd(df, colnames, filtered_query_doc, category)
            docs = ds.similar_docs(lsa, len(df.index), dist_method, n)
            doc_list = []
            dist_list = []
            for key, value in docs.items():
                doc_list.append(key)
                dist_list.append(round(value, 5))

        elif alg == "Doc2Vec":
            # word2vec
            print("########################## WORD2VEC ##########################")
            ds = D2V()
            model = ds.train(df, colnames, model_file)
            docs = ds.similar_docs(model, vecs, df, colnames, filtered_query_doc, dist_method, n)
            doc_list = []
            dist_list = []
            # for i in docs:
            #     doc_list.append(i[0])
            #     dist_list.append(round(i[1], 5))
            for key, value in docs.items():
                doc_list.append(key)
                dist_list.append(round(value, 5))

        doc_d = df.iloc[doc_list]
        doc_d["Distance"] = dist_list
        eval = Evaluate()
        doc_cats = doc_d[colnames[0]].to_list()
        score = eval.get_score(doc_cats, category)
        doc_d = doc_d.to_dict('records')

        return render_template('results.html', doc_d=doc_d, category=category, score = score)

    return render_template('index.html', form=form)

def get_vecs(vec_file, model_file):
    ds = D2V()
    model = ds.train(df, colnames, model_file)
    if not os.path.isfile(vec_file):
        vecs = ds.create_vecs(model, df, colnames)
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
df['Text cleaned'] = df.Text.apply(         #Cleaning
    lambda x: " ".join(lemma.lemmatize(re.sub(r'[^a-zA-Z]', ' ', w).lower()) for w in x.split() if
                       re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l and len(
                           re.sub(r'[^a-zA-Z]', ' ', w)) > 2))
df['Text'] = df['Text'].str.replace('/', 'FORWARD_SLASH')  # to prevent browser from getting confused

colnames.append('Text cleaned')
algorithms = ['tfidf', 'lsa', 'Doc2Vec']  # list of available methods to use
print("pre-processing done")
# pre-processing finished
#doc2vec creating vecs
model_file = 'd2v.model'
vec_file = 'vecs.txt'
vecs = get_vecs(vec_file, model_file)

if __name__ == '__main__':
    app.run(debug=True)