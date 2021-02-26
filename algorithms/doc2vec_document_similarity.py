from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import nltk
import numpy as np
import itertools
nltk.download('punkt')

class D2V:

    def __init__(self):
        pass

    def train(self, df, colnames):
        # tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(df[colnames[2]])]
        # model_d2v = Doc2Vec(vector_size=20,alpha=0.025, min_count=1, dm =1)
        # model_d2v.build_vocab(tagged_data)
        # for epoch in range(100):
        #     print('iteration {0}'.format(epoch))
        #     model_d2v.train(tagged_data,
        #                 total_examples=model_d2v.corpus_count,
        #                 epochs=model_d2v.epochs)
        # model_d2v.save("d2v.model")

        tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(df[colnames[2]])]
        model_d2v = Doc2Vec(vector_size=100, min_count=1, dm =1, epochs = 100)
        model_d2v.build_vocab(tagged_data)
        model_d2v.train(tagged_data,
                    total_examples=model_d2v.corpus_count,
                    epochs=model_d2v.epochs)
        model_d2v.save("d2v.model")

        model_d2v = Doc2Vec.load("d2v.model")
        return model_d2v

    def cosineSimilarity(self, doc, query):
        norm_doc = np.linalg.norm(doc)
        norm_query = np.linalg.norm(query)
        if norm_doc * norm_query == 0:
            return 0
        cos_theta = np.dot(doc, query) / (norm_doc * norm_query)
        return cos_theta

    def similarDocs(self, model, df, colnames, query_doc, method, amount):
        query_vector = model.infer_vector(query_doc.split())
        sims = model.docvecs.most_similar([query_vector], topn=amount)

        return sims
        # print(query_vector)
        # doc_dict = {}
        # for index, row in df.iterrows():
        #     doc_vector = model.infer_vector(row[colnames[2]].split())
        #     if method == 'cosine':
        #         val = self.cosineSimilarity(doc_vector, query_vector)
        #     doc_dict[index] = val
        #
        # doc_dict = {k: v for k, v in sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)}
        #
        # return dict(itertools.islice(doc_dict.items(), amount))
