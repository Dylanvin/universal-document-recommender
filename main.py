from algorithms.tfidf_document_similarity import TfIdf
from algorithms.doc2vec_document_similarity import D2V
from algorithms.lsa_document_similarity import LSA
from algorithms.bert_document_similarity import BERT
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from nltk.stem import WordNetLemmatizer

#pre-processing started
pd.set_option('display.max_colwidth', 200)
colnames=['Catagory', 'Text']
# df1 = pd.read_csv('datasets/20ng-test-all-terms.txt', names = colnames, sep = '\t')
# df2 = pd.read_csv('datasets/20ng-train-all-terms.txt', names = colnames, sep = '\t')
# frames = [df1, df2]
# df = pd.concat(frames)


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
colnames.append('Text cleaned')
#pre-processing finished


#call this on request
query_doc = "She is such a nice person, this sucks! She really made a huge impact as far as female billiard players go as well. I hope the best for her."
catagory = 'science'
query_doc_tokenized = word_tokenize(query_doc)
query_doc_tokenized = [token.lower() for token in query_doc_tokenized]
query_doc_tokenized = [word for word in query_doc_tokenized if not word in stopwords.words()]
filtered_query_doc = (" ").join(query_doc_tokenized)

# #TFidf
# print("########################## TFIDF ##########################")
# ds = TfIdf()
# tf_idf = ds.tdfIdf(df, colnames, filtered_query_doc, catagory)
# print("tf_idf done")
# N = 5
# method = 'cosine'
# docs = ds.similarDocs(tf_idf, len(df.index), 'cosine', N) #assumes query is the last doc in every value of key
# for key in docs:
#     print(df.iloc[key])
# print(docs)
#
# #LSA
# N = 5
# print("########################## LSA ##########################")
# ds = LSA()
# lsa = ds.tfidf_svd(df, colnames, filtered_query_doc, catagory)
# docs = ds.similarDocs(lsa, len(df.index), 'cosine', N)
# for key in docs:
#     print(df.iloc[key])
# print(docs)
#
# #word2vec
# print("########################## WORD2VEC ##########################")
# N = 5
# ds = D2V()
# model = ds.train(df, colnames)
# docs = ds.similarDocs(model, df, colnames, filtered_query_doc, 'cosine', N)
# print(docs)
# for i in range(N):
#     print(df.iloc[docs[i][0]])


print("########################BERT#####################")
ds = BERT()
docs = ds.similar_docs(df, filtered_query_doc, colnames, 'cosine', 10)
for key in docs:
    print(df.iloc[key])
print(docs)