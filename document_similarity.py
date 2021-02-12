import re
import math
import numpy as np
import itertools

class TfIdf:
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

    def tfDocAndQuery(self, df, colnames, doc, catagory):
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
        return tf, df

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
        qtf, df = self.tfDocAndQuery(df, colnames, query_doc, catagory)
        qidf = self.queryIdf(df, colnames, query_doc)
        tf_idf = {}
        for key in qidf:
            tf_idf[key] = [0] * (len(df.index))
            for i in range(len(df.index)):
                tf_idf[key][i] = qtf[key][i] * qidf[key]
        return tf_idf

    def cosineSimilarity(self, doc, query):
        norm_doc = np.linalg.norm(doc)
        norm_query = np.linalg.norm(query)
        if norm_doc * norm_query == 0:
            return 0
        cos_theta = np.dot(doc, query) / (norm_doc * norm_query)
        #theta = math.degrees(math.acos(cos_theta))
        return round(cos_theta, 10)

    def similarDocs(self, tf_idf, size, method, amount):
        if method == "cosine":
            query_tf_idf_ls = []         #tf_idf list for query doc
            for key in tf_idf:
                query_tf_idf_ls.append(tf_idf[key][size])

            doc_dict = {}
            for i in range(size):
                doc_tf_idf_ls = []
                for key in tf_idf:                                   #loop for tf_idf list of docs
                    doc_tf_idf_ls.append(tf_idf[key][i])
                angle = self.cosineSimilarity(doc_tf_idf_ls, query_tf_idf_ls)
                doc_dict[i] = angle

            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}

            return dict(itertools.islice(doc_dict.items(), amount))
