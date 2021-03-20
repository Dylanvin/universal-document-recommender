import re
import math
import itertools
from .distance import Distance
import numpy as np

class TfIdf:

    # # TF section (run before request)
    # def tf(self, df, colnames):
    #     tf = {}
    #     for index, row in df.iterrows():
    #         words = re.sub("[^\w]", " ", row[colnames[2]]).split()
    #         for word in set(words):
    #             if word in tf:
    #                 tf[word][index] = words.count(word) / len(words)
    #             else:
    #                 tf[word] = [0] * (len(df.index) + 1)
    #                 tf[word][index] = words.count(word) / len(words)
    #     return tf
    #
    # # adding query doc to TF datastructer (run after request)
    # def queryTf(self, df, tf, doc):
    #
    #     query_index = len(df.index)
    #     words = re.sub("[^\w]", " ", doc).split()
    #     for word in set(words):
    #         if word in tf:
    #             tf[word][query_index] = words.count(word) / len(words)
    #         else:
    #             tf[word] = [0] * (len(df.index) + 1)
    #             tf[word][query_index] = words.count(word) / len(words)
    #     return tf

    def tf_doc_and_query(self, df, colnames, query_doc, query_category):
        """
              Creates a TF model dict and appends query to end of df

              :param dataFrame df: corpus df
              :param list colnames: df column names
              :param str query_doc: query document
              :param str query_category: query document category
              :return dict tf AND df DataFrame:
              """
        tf = {}
        df = df.append({colnames[0]: query_category, colnames[1]: "query", colnames[2]: query_doc}, ignore_index=True)
        query_words = set(re.sub("[^\w]", " ", query_doc).split())
        text = df[colnames[2]].array
        for index in range(len(text)):
            words = re.sub("[^\w]", " ", text[index]).split()
            for word in set(words):
                if word in query_words:
                    if word in tf:
                        tf[word][index] = words.count(word) / len(words)
                    else:
                        tf[word] = [0] * (len(df.index))
                        tf[word][index] = words.count(word) / len(words)

        return tf, df

    # IDF section (run after request) only finds idf of query doc words
    def query_idf(self, df, colnames):
        """
             Creates IDF

             :param dataFrame df: corpus df
             :param list colnames: df column names
             :return dict idf:
             """
        idf = {}
        query_words = set(re.sub("[^\w]", " ", df.iloc[-1][colnames[2]]).split())
        text = df[colnames[2]].array
        for index in range(len(text)):
            words = re.sub("[^\w]", " ", text[index]).split()
            for word in set(words):
                if word in query_words:
                    if word in idf:
                        idf[word] = idf[word] + 1
                    else:
                        idf[word] = 1

        for key in idf:
            idf[key] = math.log10(float(len(df.index) / (idf[key] + 1.0)))

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
        query_words = set(re.sub("[^\w]", " ", df.iloc[-1][colnames[2]]).split())
        tf_idf = np.zeros((len(df.index), len(query_words)))
        for i, key in enumerate(qidf.keys()):
            for j in range(len(df.index)):
                tf_idf[j][i] = qtf[key][j] * qidf[key]

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

        # query_tf_idf_ls = []  # tf_idf list for query doc
        # for key in tf_idf:
        #     query_tf_idf_ls.append(tf_idf[key][size])

        dist = Distance()
        doc_dict = {}

        if method == "cosine":
            for i in range(size):
                angle = dist.cosine(tf_idf[i], tf_idf[len(tf_idf) - 1])
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}

        if method == 'euclidean':
            for i in range(size):
                angle = dist.cosine(tf_idf[i], tf_idf[len(tf_idf) - 1])
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1])}

        return dict(itertools.islice(doc_dict.items(), amount))
