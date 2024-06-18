import pickle
import pandas as pd
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity
import numpy as np

class PersistentIndex:
    def __init__(self, text_fields, keyword_fields):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.dictionary = corpora.Dictionary()
        self.tfidf = models.TfidfModel()
        self.keyword_df = None
        self.corpus = []
        self.docs = []
        self.index = None

    def fit(self, docs):
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}
        
        for field in self.text_fields:
            texts = [[word for word in doc.get(field, '').split()] for doc in docs]
            self.dictionary.add_documents(texts)
            self.corpus = [self.dictionary.doc2bow(text) for text in texts]
            self.tfidf = models.TfidfModel(self.corpus)

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)
        self.index = MatrixSimilarity(self.tfidf[self.corpus])

    def save(self, path):
        with open(f'{path}_corpus.pkl', 'wb') as f:
            pickle.dump(self.corpus, f)
        self.dictionary.save(f'{path}_dictionary.gensim')
        self.tfidf.save(f'{path}_tfidf_model.gensim')
        with open(f'{path}_docs.pkl', 'wb') as f:
            pickle.dump(self.docs, f)
        self.keyword_df.to_pickle(f'{path}_keyword_df.pkl')
        self.index.save(f'{path}_index.gensim')

    def load(self, path):
        with open(f'{path}_corpus.pkl', 'rb') as f:
            self.corpus = pickle.load(f)
        self.dictionary = corpora.Dictionary.load(f'{path}_dictionary.gensim')
        self.tfidf = models.TfidfModel.load(f'{path}_tfidf_model.gensim')
        with open(f'{path}_docs.pkl', 'rb') as f:
            self.docs = pickle.load(f)
        self.keyword_df = pd.read_pickle(f'{path}_keyword_df.pkl')
        self.index = MatrixSimilarity.load(f'{path}_index.gensim')

    def update(self, new_docs):
        self.docs.extend(new_docs)
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            new_texts = [[word for word in doc.get(field, '').split()] for doc in new_docs]
            self.dictionary.add_documents(new_texts)
            new_corpus = [self.dictionary.doc2bow(text) for text in new_texts]
            self.corpus.extend(new_corpus)
            self.tfidf.add_documents(new_corpus)

        for doc in new_docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        new_keyword_df = pd.DataFrame(keyword_data)
        self.keyword_df = pd.concat([self.keyword_df, new_keyword_df], ignore_index=True)
        self.index = MatrixSimilarity(self.tfidf[self.corpus])

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        query_texts = [[word for word in query.split()]]
        query_bow = [self.dictionary.doc2bow(text) for text in query_texts]
        query_tfidf = self.tfidf[query_bow[0]]

        scores = np.array(self.index[query_tfidf])

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]

        return top_docs
