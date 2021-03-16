from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .distance import Distance
import itertools


class LSA:

    def tfidf_svd(self, df, colnames, query_doc, query_category):
        """
        Creates an LSA model from given documents

        :param dataFrame df: corpus df
        :param list colnames: df column names
        :param str query_doc: query document
        :param str query_category: query document category
        :return: LSA model
        """
        df = df.append({colnames[0]: query_category, colnames[1]: "query", colnames[2]: query_doc}, ignore_index=True)
        vectorizer = TfidfVectorizer()
        bow = vectorizer.fit_transform(df[colnames[2]])
        svd = TruncatedSVD(n_components=20)
        lsa = svd.fit_transform(bow)
        return lsa

    def similar_docs(self, lsa, size, method, amount):
        """
        returns N documents which are most similar in ascending order

        :param lsa: LSA model of corpus
        :param size: size of corpus (in order to exclude last entry which is the query)
        :param str method: measurement method. Either 'cosine' or 'euclidean'
        :param int amount: amount of documents to be returned
        :return dict: dict documents in format {doc id:measurement}
        """
        doc_dict = {}
        dist = Distance()
        if method == 'cosine':
            for i in range(size):
                angle = dist.cosine(lsa[i], lsa[len(lsa) - 1])
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}

        if method == 'euclidean':
            for i in range(size):
                angle = dist.euclidean(lsa[i], lsa[len(lsa) - 1])
                doc_dict[i] = angle
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1])}

        return dict(itertools.islice(doc_dict.items(), amount))
