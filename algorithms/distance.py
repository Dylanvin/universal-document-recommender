import numpy as np

class Distance:
    def cosine(self, doc, query):
        norm_doc = np.linalg.norm(doc)
        norm_query = np.linalg.norm(query)
        if norm_doc * norm_query == 0:
            return 0
        cos_theta = np.dot(doc, query) / (norm_doc * norm_query)
        return cos_theta

    def euclidean(self, doc, query):
        return np.linalg.norm(np.array(doc) - np.array(query))