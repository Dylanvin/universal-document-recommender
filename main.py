from document_methods import DocumentSimilarity
import pandas as pd

colnames=['Catagory', 'Text']
df = pd.read_csv('20ng-test-all-terms.txt', names = colnames, sep = '\t')
query_doc = "data science is the sexiest job of the 21st century"

ds = DocumentSimilarity()
tf = ds.tf(df, colnames)
qtf = dict(tf)

#call this on request
tf_idf = ds.tdfIdf(tf, df, colnames, query_doc)

print(tf_idf)
