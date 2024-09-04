import numpy as np
import pandas as pd
import time
from collections import OrderedDict
import pickle as pk
import glob

from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, dot, Reshape, Dropout, Flatten, Conv1D, MaxPooling1D, Activation, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot, Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils 
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L1L2
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
EMBEDDING_DIM = 300


# print(len(embedding_matrix))

embedding_layer = Embedding(vocab_size,
                            EMBEDDING_DIM,
                            input_length=max_length,
                            trainable=False)

bias_regularizer = [L1L2(l1=0.01, l2=0.0),L1L2(l1=0.01, l2=0.01)]

# t_name = “{}-{}-{}“.format(---embedding type name-----,time())
# tensorboard = TensorBoard(log_dir=“lasttest_logs/{}“.format(t_name))




model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))

model.add(Conv1D(filters=80, kernel_size = 3))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))


model.add(LSTM(64,dropout=0.2,recurrent_dropout=0.2, bias_regularizer = bias_regularizer[1]))

model.add(Activation("tanh"))


model.add(Dense(1, activation="sigmoid"))


model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.save('lstm_l2.keras')
print(model.summary())
