from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance
import numpy as np
import itertools

class LSA:
    def __init__(self):
        pass

    def cosineSimilarity(self, doc, query):
        norm_doc = np.linalg.norm(doc)
        norm_query = np.linalg.norm(query)
        if norm_doc * norm_query == 0:
            return 0
        cos_theta = np.dot(doc, query) / (norm_doc * norm_query)
        return cos_theta

    def tfidf_svd(self, df, colnames, doc, category):
        df = df.append({colnames[0]:category, colnames[1]:"query", colnames[2]:doc}, ignore_index=True)
        vectorizer = TfidfVectorizer()
        bow = vectorizer.fit_transform(df[colnames[2]])
        print("bow done")
        svd = TruncatedSVD(n_components=20)
        lsa = svd.fit_transform(bow)
        print("svd done")
        return lsa

    def similarDocs(self, lsa, size, method, amount):
        doc_dict = {}
        for i in range(size):
            angle = self.cosineSimilarity(lsa[i], lsa[len(lsa) - 1])
            doc_dict[i] = angle

        doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}
        return dict(itertools.islice(doc_dict.items(), amount))
