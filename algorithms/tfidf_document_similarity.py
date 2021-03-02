import re
import math
import itertools
from .distance import Distance


class TfIdf:
    def __init__(self):
        pass

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

    def tfDocAndQuery(self, df, colnames, query_doc, query_category):
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
        for index, row in df.iterrows():
            words = re.sub("[^\w]", " ", row[colnames[2]]).split()
            for word in set(words):
                if word in query_words:
                    if word in tf:
                        tf[word][index] = words.count(word) / len(words)
                    else:
                        tf[word] = [0] * (len(df.index))
                        tf[word][index] = words.count(word) / len(words)
        return tf, df

    # IDF section (run after request) only finds idf of query doc words
    def queryIdf(self, df, colnames):
        """
             Creates IDF

             :param dataFrame df: corpus df
             :param list colnames: df column names
             :return dict idf:
             """
        idf = {}
        query_words = set(re.sub("[^\w]", " ", df.iloc[-1][colnames[2]]).split())
        for index, row in df.iterrows():
            words = re.sub("[^\w]", " ", row[colnames[2]]).split()
            for word in set(words):
                if word in query_words:
                    if word in idf:
                        idf[word] = idf[word] + 1
                    else:
                        idf[word] = 1

        for key in idf:
            idf[key] = math.log10(float((len(df.index)) / (idf[key] + 1.0)))

        return idf

    def tdfIdf(self, df, colnames, query_doc, query_category):
        """
           Preforms TF * IDF

           :param dataFrame df: corpus df
           :param list colnames: df column names
           :param str query_doc: query document
           :param str query_category: query document category
           :return dict tf_idf:
           """
        qtf, df = self.tfDocAndQuery(df, colnames, query_doc, query_category)
        qidf = self.queryIdf(df, colnames)
        tf_idf = {}
        for key in qidf:
            tf_idf[key] = [0] * (len(df.index))
            for i in range(len(df.index)):
                tf_idf[key][i] = qtf[key][i] * qidf[key]
        return tf_idf

    def similarDocs(self, tf_idf, size, method, amount):
        """
              returns N documents which are most similar in ascending order to last document (assumed to be query)
              in tf_idf dict

              :param dict tf_idf: tf_idf model
              :param int size: amount of docs
              :param str method:
              :param int amount:
              :return:
              """

        query_tf_idf_ls = []  # tf_idf list for query doc
        for key in tf_idf:
            query_tf_idf_ls.append(tf_idf[key][size])

        dist = Distance()
        doc_dict = {}

        if method == "cosine":
            for i in range(size):
                doc_tf_idf_ls = []
                for key in tf_idf:  # loop for tf_idf list of docs
                    doc_tf_idf_ls.append(tf_idf[key][i])
                angle = dist.cosine(doc_tf_idf_ls, query_tf_idf_ls)
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}

        if method == 'euclidean':
            for i in range(size):
                doc_tf_idf_ls = []
                for key in tf_idf:  # loop for tf_idf list of docs
                    doc_tf_idf_ls.append(tf_idf[key][i])
                angle = dist.euclidean(doc_tf_idf_ls, query_tf_idf_ls)
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1])}




        return dict(itertools.islice(doc_dict.items(), amount))
