from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .distance import Distance
import itertools

class LSA:
    def __init__(self):
        pass

    def tfidf_svd(self, df, colnames, doc, category):
        df = df.append({colnames[0]:category, colnames[1]:"query", colnames[2]:doc}, ignore_index=True)
        vectorizer = TfidfVectorizer()
        bow = vectorizer.fit_transform(df[colnames[2]])
        svd = TruncatedSVD(n_components=20)
        lsa = svd.fit_transform(bow)
        return lsa

    def similar_docs(self, lsa, size, method, amount):
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
