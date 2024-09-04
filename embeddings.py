import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import time
import pickle as pk

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

class Embeddings:

    def __init__(self):
        self.embeddings = None
        self.vocab = None
        self.dim = None

    def set_embeddings(self, embeddings, vocab):
        """Embeddings should be (dim, n) shaped"""
        self.embeddings = embeddings
        self.vocab = np.array(vocab)

    def generate_word2vec_embeddings(self, sentences, size, workers=4, min_count=1, **kwargs):
        model = Word2Vec(reviews_tokenized, size=size, workers=workers, min_count=min_count, **kwargs)
        self.embeddings = model.vectors.T
        self.vocab = np.array(model.index2word)

    def load_w2v_embeddings(self, fname, binary):
        model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=binary)
        self.embeddings = model.vectors.T
        self.vocab = np.array(model.index2word)

    def trim_embeddings(self, words):
        pass

    def save_embeddings(self, fname):
        pk.dump((self.vocab, self.embeddings), open(fname, 'wb'), protocol=4)

    def load_embeddings(self, fname):
        self.vocab, self.embeddings = pk.load(open(fname, 'rb'))

    def _compute_avg_noise(self, words, neighbors, lexicon, k=10):
        translator = np.vectorize(lexicon.get)
        word_sentiments = translator(words) >= 5
        neighbor_sentiments = translator(neighbors) >= 5

        noise = (word_sentiments[:,None] != neighbor_sentiments).sum()
        noise /= (k * len(words))

        return noise

    def most_similar(self, word, topn=10):
        try:
            idx = np.where(self.vocab == word)[0][0]
        except KeyError:
            print('Word not in vocabulary!')
            return []
        subs = self.embeddings - self.embeddings[:,[idx]]
        out = np.sqrt(np.einsum('ij,ij->j',subs,subs))
        idx = out.argsort()[1:topn+1]
        return [(self.vocab[i], out[i]) for i in idx]

    def _sort_by_sentiment(self, words, neighbors, lexicon):
        translator = np.vectorize(lexicon.get)
        word_sentiments = translator(words)
        neighbor_sentiments = translator(neighbors)

        return abs(word_sentiments[:,None] - neighbor_sentiments).argsort(1)

    def refine_embeddings(self, lexicon, max_iter=3, k=10, alpha=1, beta=10, calc_noise=False, verbose=False):
        assert self.embeddings is not None
        self.noise_list = []

        idx = [i for i,word in enumerate(self.vocab) if word in lexicon]
        refined_words = self.vocab[idx]
        refined_embeddings = self.embeddings[:,idx]

        if verbose:
            print('Computing euclidean distances...')
        dist = euclidean_distances(refined_embeddings.T, refined_embeddings.T, squared=True)
        nearest = np.argsort(dist)[:, 1:k + 1]

        if calc_noise:
            noise = self._compute_avg_noise(refined_words, refined_words[nearest], lexicon, k)
            self.noise_list.append(noise)
            print('Noise: {:.4f}'.format(noise))

        if verbose:
            print('Sorting {} nearest neighbors by sentiment...'.format(k))
        sorted_nearest_neighbors = self._sort_by_sentiment(refined_words, refined_words[nearest], lexicon)

        row_idx = np.repeat(np.arange(0, dist.shape[0]), [k]).reshape(dist.shape[0], k)
        comb_sort = nearest[row_idx, sorted_nearest_neighbors]

        if verbose:
            print('Refining embeddings...')
        W = np.zeros(dist.shape)
        W[row_idx, comb_sort] = [1/i for i in range(1, k+1)]

        for i in range(max_iter):
            if verbose:
                print('Iteration {} of {}'.format(i + 1, max_iter))
            M = alpha * refined_embeddings + beta * refined_embeddings@W.T
            b = alpha + beta * W.sum(axis=1)
            refined_embeddings = M / b

            # print(refined_embeddings[:,0])
            if calc_noise:
                dist = euclidean_distances(refined_embeddings.T, refined_embeddings.T, squared=True)
                nearest = np.argsort(dist)[:, 1:k + 1]
                noise = self._compute_avg_noise(refined_words, refined_words[nearest], lexicon, k)
                self.noise_list.append(noise)
                print('Noise: {:.4f}'.format(noise))

            if verbose:
                print('-------------------')

        self.embeddings[:,idx] = refined_embeddings


if __name__ == "__main__":

    # data = pd.read_csv('../../data/phase1_movie_reviews-train.csv')
    # reviews = data["reviewText"].astype(str).values
    # reviews_tokenized = [text_to_word_sequence(doc) for doc in reviews]
    # pk.dump(reviews_tokenized, open('reviews_tokenized.pk','wb'))
    reviews_tokenized = pk.load(open('reviews_tokenized.pk','rb'))

    lexicon = pd.read_csv('../../data/BRM-emot-submit.csv', usecols=['Word','V.Mean.Sum'], index_col=0)
    lexicon = lexicon['V.Mean.Sum'].to_dict()

    embeddings = Embeddings()
    # embeddings.load_w2v_embeddings("../../data/word2vec_google/GoogleNews-vectors-negative300.bin", binary=True)
    embeddings.load_w2v_embeddings("../../data/glove_w2v.840B.300d.txt", binary=False)
    # embeddings.generate_word2vec_embeddings(reviews_tokenized, size=100, workers=8)
    embeddings.save_embeddings('models/glove_org.emb')
    # embeddings.load_embeddings('models/word2vec_org.emb')
    # print(embeddings.most_similar('bad'))
    # embeddings.refine_embeddings(lexicon, max_iter=10, k=10,  alpha=1, beta=beta, calc_noise=True, verbose=True)
    # print(embeddings.most_similar('bad'))
    # exit()
    beta = 0.1
    max_iter = 1
    for k in [5, 10, 15]:
        print('k: ', k)
        embeddings.load_embeddings('models/glove_org.emb')
        embeddings.refine_embeddings(lexicon, max_iter=max_iter, k=k,  alpha=1, beta=beta, calc_noise=True, verbose=True)
        embeddings.save_embeddings('models/glove_beta_{}_iter_{}_k_{}.emb'.format(beta, max_iter, k))
        # print(embeddings.most_similar('bad'))
        print('-----------')
