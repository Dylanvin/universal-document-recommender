from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from .distance import Distance
from nltk.tokenize import word_tokenize
import os.path
import nltk
import itertools

nltk.download('punkt')

class D2V:

    def __init__(self):
        pass

    def train(self, df, colnames, file):
        """
        Trains a Doc2Vec model if it does not a;ready exist, otherwise loads pre-existing file

        :param DataFrame df: corpus df
        :param list colnames: df column names
        :param str file: path fo model
        :return: doc2vec model
        """
        if not os.path.isfile(file):
            tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(df[colnames[2]])]
            model_d2v = Doc2Vec(vector_size=100, min_count=1, dm =1, epochs = 100)
            model_d2v.build_vocab(tagged_data)
            model_d2v.train(tagged_data,
                        total_examples=model_d2v.corpus_count,
                        epochs=model_d2v.epochs)
            model_d2v.save(file)
        else:
            model_d2v = Doc2Vec.load(file)
        return model_d2v


    def similar_docs(self, model, vecs, query_doc, method, amount):
        """
        returns N documents which are most similar in ascending order

        :param model: Doc2Vec model
        :param list[list] vecs: vector representation of each document in corpus
        :param str query_doc: query document
        :param str method: measurement method. Either 'cosine' or 'euclidean'
        :param amount: amount of documents to be returned
        :return dict: dict documents in format {doc id:measurement}
        """
        # query_vector = model.infer_vector(query_doc.split())
        # sims = model.docvecs.most_similar([query_vector], topn=amount)
        # return sims
        dist = Distance()
        query_vector = model.infer_vector(query_doc.split())
        doc_dict = {}

        # if method == 'cosine':
        #     for index, row in df.iterrows():
        #         doc_vector = model.infer_vector(row[colnames[2]].split())
        #         val = dist.cosine(doc_vector, query_vector)
        #         doc_dict[index] = val
        #     doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}
        #
        # elif method == 'euclidean':
        #     for index, row in df.iterrows():
        #         doc_vector = model.infer_vector(row[colnames[2]].split())
        #         val = dist.euclidean(doc_vector, query_vector)
        #         doc_dict[index] = val
        #     doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1])}

        if method == 'cosine':
            for i in range(len(vecs)):
                val = dist.cosine(vecs[i], query_vector)
                doc_dict[i] = val
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}

        elif method == 'euclidean':
            for i in range(len(vecs)):
                val = dist.euclidean(vecs[i], query_vector)
                doc_dict[i] = val
            doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1])}

        return dict(itertools.islice(doc_dict.items(), amount))

    def create_vecs(self, model, df, colnames):
        """
        converts input documents into vectors

        :param model: Doc2Vec model
        :param DataFrame df: corpus df
        :param list colnames: df column names
        :return list[list]: vectorised documents
        """
        vecs = []
        for index, row in df.iterrows():
            vecs.append(model.infer_vector(row[colnames[2]].split()))

        return vecs