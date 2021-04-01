import numpy as np


class Distance:
    def cosine(self, doc, query):
        """
        finds cosine similarity of two vectors.

        :param list doc: vectorised document
        :param list query: vectorised query document
        :return float: similarity value between 0 (no similarity) and 1 (complete similarity)
        """
        norm_doc = np.linalg.norm(doc)
        norm_query = np.linalg.norm(query)
        if norm_doc * norm_query == 0:
            return 0
        cos_theta = np.dot(doc, query) / (norm_doc * norm_query)
        return cos_theta

    def euclidean(self, doc, query):
        """
        finds euclidean distance of two vectors.

        :param list doc: vectorised document
        :param list query: vectorised query document
        :return float: the smaller the value the more similar the documents
        """
        return np.linalg.norm(np.array(doc) - np.array(query))
