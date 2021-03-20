import re
import math
import numpy as np

class test:
    def tf_doc_and_query(self, df, colnames, query_doc, query_category):
        """
              Creates a TF model dict and appends query to end of df

              :param dataFrame df: corpus df
              :param list colnames: df column names
              :param str query_doc: query document
              :param str query_category: query document category
              :return dict tf AND df DataFrame:
              """

        df = df.append({colnames[0]: query_category, colnames[1]: "query", colnames[2]: query_doc}, ignore_index=True)
        query_words = set(re.sub("[^\w]", " ", query_doc).split())
        tf = np.zeros((len(df.index), len(query_words)))
        i = 0
        text = df[colnames[2]].array
        for word in query_words:
            for index in range(len(text)):
                words = re.sub("[^\w]", " ", text[index]).split()
                if word in words:
                    if index in tf:
                        tf[index][i] = words.count(word) / len(words)
                    else:
                        tf[index] = np.zeros(len(query_words))
                        tf[index][i] = words.count(word) / len(words)
            i = i + 1
        print("tf done")
        return tf, df


    # IDF section (run after request) only finds idf of query doc words
    def query_idf(self, df, colnames):
        """
             Creates IDF

             :param dataFrame df: corpus df
             :param list colnames: df column names
             :return dict idf:
             """
        idf = []
        query_words = set(re.sub("[^\w]", " ", df.iloc[-1][colnames[2]]).split())
        for word in query_words:
            num = 0
            for index, row in df.iterrows():
                words = set(re.sub("[^\w]", " ", row[colnames[2]]).split())
                if word in words:
                    num = num + 1
            idf.append(num)

        idf[:] = [math.log10(float(len(df.index) / (x + 1.0))) for x in idf]
        idf = np.array(idf)

        return idf


    def tf_idf(self, df, colnames, query_doc, query_category):
        """
           Preforms TF * IDF

           :param dataFrame df: corpus df
           :param list colnames: df column names
           :param str query_doc: query document
           :param str query_category: query document category
           :return dict tf_idf:
           """
        qtf, df = self.tf_doc_and_query(df, colnames, query_doc, query_category)
        qidf = self.query_idf(df, colnames)
        tf_idf = []
        i = 0
        for key, value in qtf.items():
            tf_idf.append(value * qidf[i])
            i = i + 1
        tf_idf = np.array(tf_idf)
        return tf_idf


    def similar_docs(self, tf_idf, size, method, amount):
        """
              returns N documents which are most similar in ascending order to last document (assumed to be query)
              in tf_idf dict

              :param dict tf_idf: tf_idf model
              :param int size: amount of docs
              :param str method: measurement method. Either 'cosine' or 'euclidean'
              :param int amount: amount of documents to be returned
              :return dict: dict documents in format {doc id:measurement}
              """

        doc_dict = {}
        dist = Distance()
        if method == 'cosine':
            for i in range(size):
                angle = dist.cosine(tf_idf[i], tf_idf[len(tf_idf) - 1])
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}

        if method == 'euclidean':
            for i in range(size):
                angle = dist.euclidean(tf_idf[i], tf_idf[len(tf_idf) - 1])
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1])}

        return dict(itertools.islice(doc_dict.items(), amount))
