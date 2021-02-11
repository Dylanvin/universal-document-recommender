import re
import math

class DocumentSimilarity:
    def __init__(self):
        pass

    #TF section (run before request)
    def tf(self, df, colnames):
        tf = {}
        for index, row in df.iterrows():
            words = re.sub("[^\w]", " ", row[colnames[1]]).split()
            for word in set(words):
                if word in tf:
                    tf[word][index] = words.count(word)/len(words)
                else:
                    tf[word] = [0] * (len(df.index) + 1)
                    tf[word][index] = words.count(word)/len(words)
        return tf

    #adding query doc to TF datastructer (run after request)
    def queryTf(self, df, tf, doc):
        query_index = len(df.index)
        words = re.sub("[^\w]", " ", doc).split()
        for word in set(words):
            if word in tf:
                tf[word][query_index] = words.count(word)/len(words)
            else:
                tf[word] = [0] * (len(df.index) + 1)
                tf[word][query_index] = words.count(word)/len(words)
        return tf

    def tfAndQuery(self, df, colnames, doc, catagory):
        tf = {}
        df = df.append({colnames[0]:catagory, colnames[1]:doc}, ignore_index=True)
        query_words = set(re.sub("[^\w]", " ", doc).split())
        for index, row in df.iterrows():
            words = re.sub("[^\w]", " ", row[colnames[1]]).split()
            for word in set(words):
                if word in query_words:
                    if word in tf:
                        tf[word][index] = words.count(word)/len(words)
                    else:
                        tf[word] = [0] * (len(df.index))
                        tf[word][index] = words.count(word)/len(words)
        return tf

    #IDF section (run after request) only finds idf of query doc words
    def queryIdf(self, df, colnames, doc):
        idf = {}
        query_words = set(re.sub("[^\w]", " ", doc).split())
        for index, row in df.iterrows():
            words = re.sub("[^\w]", " ", row[colnames[1]]).split()
            for word in set(words):
                if word in query_words:
                    if word in idf:
                        idf[word] = idf[word] + 1
                    else:
                        idf[word] = 1

        for key in idf:
            idf[key] = math.log10(float((len(df.index))/(idf[key] + 1.0)))

        return idf

    def tdfIdf(self, df, colnames, query_doc, catagory):
        qtf = self.tfAndQuery(df, colnames, query_doc, catagory)
        qidf = self.queryIdf(df, colnames, query_doc)
        tf_idf = {}
        for key in qidf:
            tf_idf[key] = [0] * (len(df.index) + 1)
            for i in range(len(df.index) + 1):
                tf_idf[key][i] = qtf[key][i] * qidf[key]
        return tf_idf
