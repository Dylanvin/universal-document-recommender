from sentence_transformers import SentenceTransformer
from .distance import Distance
import itertools


class BERT:
    def similar_docs(self, document_vecs, query, method, amount):
        """
        Uses the BERT model to vectorise query document and to match it against supplies vectorised corpus list,
        returns N documents which are most similar in ascending order

        :param list[list] document_vecs: list of lists containing vectorised representation of corpus
        :param str query: query document
        :param str method: measurement method. Either 'cosine' or 'euclidean'
        :param amount: amount of documents to be returned
        :return dict: dict documents in format {doc id:measurement}
        """
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        query_vec = sbert_model.encode(query)
        dist = Distance()
        doc_dict = {}

        if method == 'cosine':
            for i in range(len(document_vecs)):
                val = dist.cosine(document_vecs[i], query_vec)
                doc_dict[i] = val
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}

        elif method == 'euclidean':
            for i in range(len(document_vecs)):
                val = dist.euclidean(document_vecs[i], query_vec)
                doc_dict[i] = val
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1])}

        return dict(itertools.islice(doc_dict.items(), amount))

    def create_vecs(self, df, colnames):
        """
        converts input documents into vectors

        :param DataFrame df: corpus df
        :param list colnames: df column names
        :return list[list]: vectorised documents
        """
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        document_vecs = sbert_model.encode(df[colnames[2]])

        return document_vecs
