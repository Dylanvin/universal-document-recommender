from sentence_transformers import SentenceTransformer
from .distance import Distance
import itertools
from transformers import AutoTokenizer, AutoModel
import torch


class BERT:

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

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

        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        # model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        #
        # encoded_input = tokenizer(query, padding=True, truncation=True, max_length=128, return_tensors='pt')
        #
        # with torch.no_grad():
        #     model_output = model(**encoded_input)
        #
        # query_vec = self.mean_pooling(model_output, encoded_input['attention_mask'])

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

        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        # model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        #
        # encoded_input = tokenizer(df[colnames[2]], padding=True, truncation=True, max_length=128, return_tensors='pt')
        #
        # with torch.no_grad():
        #     model_output = model(**encoded_input)
        #
        # document_vecs = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return document_vecs
