import numpy as np
import pandas as pd
import time
from collections import OrderedDict
import pickle as pk
import glob

from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, dot, Reshape, Dropout, Flatten, Conv1D, MaxPooling1D, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot, Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils 
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from IPython.display import SVG

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.sparse import coo_matrix, hstack, csr_matrix

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

# fname = 'w2v_beta_5.emb'
results = dict()
for fname in glob.glob('models/gl_w2v*_k_15.emb'):
    print("Using ", fname)
    vocab, embeddings = pk.load(open(fname, 'rb'))
    vocab = {word:i for i, word in enumerate(vocab)}

    EMBEDDING_DIM = embeddings.shape[0]

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    embedding_idx = []
    word_idx = []
    for word, i in word_index.items():
        try:
            embedding_idx.append(vocab[word])
            word_idx.append(i)
            # embedding_vector = embeddings[:,vocab[word]]
        except KeyError:
            continue
        # embedding_matrix[i] = embedding_vector
    print("Filtering embeddings...")
    embeddings = embeddings[:,embedding_idx]
    embedding_matrix[word_idx] = embeddings.T

    # print(len(embedding_matrix))

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.25))

    model.add(Conv1D(filters=40, kernel_size=3))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Activation("relu"))


    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=optimizers.adam(lr=0.001), metrics=["accuracy"])
    # print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min')

    hist = model.fit(X_rev_train, y_rev_train, epochs=20, batch_size=32, validation_data=(X_rev_dev,y_rev_dev),callbacks=[early_stopping,reduce_lr],verbose=1)
    best = max(hist.history['val_acc'])
    print('Best validation accuracy: ',best)
    results[fname] = best
    # pk.dump(model.predict(X_rev_dev), open('predictions.pk','wb'))
    # pk.dump(model.predict_proba(X_rev_dev), open('predictions_proba.pk','wb'))

    # break

for fname, score in results.items():
    print('{}\t{}'.format(fname,score))
