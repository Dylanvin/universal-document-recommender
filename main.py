from document_methods import DocumentSimilarity
import pandas as pd

colnames=['Catagory', 'Text']
df = pd.read_csv('20ng-test-all-terms.txt', names = colnames, sep = '\t')
query_doc = "data science is the sexiest job of the 21st century"
catagory = 'science'

ds = DocumentSimilarity()
#tf = ds.tf(df, colnames)
print("tf done")


#call this on request
#qtf = dict(tf)
tf_idf = ds.tdfIdf(df, colnames, query_doc, catagory)
print("tf_idf done")
print(tf_idf)
