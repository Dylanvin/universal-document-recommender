from sentence_transformers import SentenceTransformer
from .distance import Distance
import itertools

class BERT:
    def similar_docs(self, document_vecs, query, method, amount):
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
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        document_vecs = sbert_model.encode(df[colnames[2]])

        return document_vecs