import numpy as np
import pandas as pd
import time
from collections import OrderedDict
import pickle as pk
import glob

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, dot, Reshape, Dropout, Flatten, Conv1D, MaxPooling1D, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot, Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils 
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from IPython.display import SVG
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.sparse import coo_matrix, hstack, csr_matrix

class NNModel:

    def __init__(self, model_fname, vocab_size, emb_fname=None):
        self.emb_fname = emb_fname
        self.model = None
        self.model_fname = model_fname
        self.vocab_size = vocab_size

    def _change_embedding(self, embedding_layer):
        layers = [l for l in self.model.layers]

        x = layers[0].output
        for i in range(1, len(layers)):
            if i == layer_id:
                x = embedding_layer(x)
            else:
                x = layers[i](x)

        new_model = Model(input=layers[0].input, output=x)
        return new_model



    def _build_model(self, word_index, input_length):
        self.model = load_model(self.model_fname)

        if self.emb_fname is not None:
            vocab, embeddings = pk.load(open(self.emb_fname, 'rb'))
            vocab = {word:i for i, word in enumerate(vocab)}

            emb_dim = embeddings.shape[0]

            embedding_matrix = np.zeros((self.vocab_size, emb_dim))
            embedding_idx = []
            word_idx = []
            for word, i in word_index.items():
                if i > self.vocab_size - 1:
                    continue
                try:
                    embedding_idx.append(vocab[word])
                    word_idx.append(i)
                except KeyError:
                    continue
            print("Filtering embeddings...")
            embeddings = embeddings[:,embedding_idx]
            embedding_matrix[word_idx] = embeddings.T

            embedding_layer = Embedding(self.vocab_size,
                                        emb_dim,
                                        weights=[embedding_matrix],
                                        input_length=max_length,
                                        trainable=False)
            self.model.layers[0] = embedding_layer

    def fit(self, X, y, word_index, X_dev=None, y_dev=None):
        self._build_model(word_index, X.shape[1])

        if X_dev is not None and y_dev is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min')
            checkpoint = ModelCheckpoint(self.model_fname+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')


            hist = self.model.fit(X_rev_train, y_rev_train, epochs=20,
                                                       batch_size=32, 
                                                       validation_data=(X_rev_dev,y_rev_dev),
                                                       callbacks=[ModelCheckpoint,early_stopping,reduce_lr],
                                                       verbose=1)

            self.model = load_model(self.model_fname + '.hdf5')

        else:
            hist = self.model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

if __name__ == "__main__":
    vocab_size = 30000
    max_length = 500

    # data = pd.read_csv('../../data/phase1_movie_reviews-train.csv')
    # reviews = data["reviewText"].astype(str).values
    #
    # lb = preprocessing.LabelBinarizer()
    # y = lb.fit_transform(data["polarity"])
    # y = np.array([yi[0] for yi in y])
    #
    # X_rev_train, X_rev_dev, y_rev_train, y_rev_dev = train_test_split(reviews, y, test_size=0.33, random_state=50)
    #
    # tokenizer = Tokenizer(num_words=vocab_size)
    # tokenizer.fit_on_texts(X_rev_train)
    # word_index = tokenizer.word_index
    #
    # X_rev_train = tokenizer.texts_to_sequences(X_rev_train)
    # X_rev_dev = tokenizer.texts_to_sequences(X_rev_dev)
    #
    # X_rev_train = pad_sequences(X_rev_train, maxlen=max_length)
    # X_rev_dev = pad_sequences(X_rev_dev, maxlen=max_length)
    #
    #
    # pk.dump((X_rev_train, X_rev_dev, y_rev_train, y_rev_dev, word_index), open('dev_split.pk','wb'))
    X_rev_train, X_rev_dev, y_rev_train, y_rev_dev, word_index = pk.load(open('dev_split.pk','rb'))
    model = NNModel(model_fname="lstm_l1.keras",emb_fname='models/gl_w2v_org.emb', vocab_size=vocab_size)
    model.fit(X_rev_train, y_rev_train, word_index, X_rev_dev, y_rev_dev)
    model.predict(X_rev_train)
    print(model.predict(X_rev_dev))
    print(model.evaluate(X_rev_dev, y_rev_dev))
