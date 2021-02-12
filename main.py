from document_similarity import TfIdf
import pandas as pd

pd.set_option('display.max_colwidth', 100)
colnames=['Catagory', 'Text']
df = pd.read_csv('20ng-test-all-terms.txt', names = colnames, sep = '\t')
query_doc = "my bike broke and now its leaking"
catagory = 'science'
ds = TfIdf()


#call this on request
tf_idf = ds.tdfIdf(df, colnames, query_doc, catagory)
print("tf_idf done")

N = 5
method = 'cosine'
docs = ds.similarDocs(tf_idf, len(df.index), 'cosine', N) #assumes query is the last doc in every value of key


for key in docs:
    print(df.iloc[key])
