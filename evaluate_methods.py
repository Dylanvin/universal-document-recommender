from algorithms.tfidf_document_similarity import TfIdf
from algorithms.doc2vec_document_similarity import D2V
from algorithms.lsa_document_similarity import LSA
from algorithms.bert_document_similarity import BERT
from algorithms.evaluate import Evaluate
import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.datasets import fetch_20newsgroups
from nltk.stem import WordNetLemmatizer
import os
import pickle
import numpy as np
import html2text
import requests


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


def run_algs(alg, dist_method, n, category, filtered_query_doc):

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

    return score

algorithms = ['TfIdf', 'LSA', 'Doc2Vec', 'BERT']  # list of available methods to use
mes = ['cosine','euclidean']
test_docs_df = pd.read_csv('test_docs.txt', delimiter = ",")
print(test_docs_df.columns.values)
     
    

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True

test_docs_df["Text"] = test_docs_df.Url.apply(
    lambda x: h.handle(requests.get(x).text)) 

test_docs_df['Text_cleaned'] = test_docs_df.Text.apply(  # Cleaning
    lambda x: " ".join(lemma.lemmatize(re.sub(r'[^a-zA-Z]', ' ', w).lower()) for w in x.split() if
                       re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l and len(
                           re.sub(r'[^a-zA-Z]', ' ', w)) > 2))


#print("running test")
#for alg in algorithms:
#    for m in mes:
#        test_docs_df[alg + "(" + m + ")"] = test_docs_df.apply(
#             lambda x: run_algs(alg, m, 5, x.Category, x.Text_cleaned), axis=1)
#
#print("test complete")
#test_docs_df = test_docs_df.drop(columns = ["Text", "Text_cleaned", "Url"], axis=1)
#test_docs_df.to_csv('results.csv', index=False)

results_df = pd.read_csv('results.csv', delimiter = ",")
print(results_df.mean(axis=0))