import re
import numpy as np
import pandas as pd
from time import time


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

from scipy.sparse import hstack, csr_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten, LSTM, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau




data = pd.read_csv('../../data/phase1_movie_reviews-train.csv')

year = pd.get_dummies(data["year"])
summaries = data["summary"].astype(str).values
reviews = data["reviewText"].astype(str).values

clean_summaries = np.array([text_to_word_sequence(line) for line in summaries])
clean_reviews = np.array([text_to_word_sequence(line) for line in reviews])

vocab_size = 10000
max_sentence = 500

# For cleaned texts
summaries_mat = [one_hot(" ".join(line), vocab_size) for line in clean_summaries]
reviews_mat = [one_hot(" ".join(line), vocab_size) for line in clean_reviews]

X_sum = pad_sequences(summaries_mat, maxlen=max_sentence, padding='pre')
X_rev = pad_sequences(reviews_mat, maxlen=max_sentence, padding='pre')


X_letout = csr_matrix(hstack([X_sum,X_rev,year]))
lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(data["polarity"])
y = np.array([yi[0] for yi in y])


X_sum_train, X_sum_dev, y_sum_train, y_sum_dev = train_test_split(X_sum, y, test_size=0.33, random_state=50)
X_rev_train, X_rev_dev, y_rev_train, y_rev_dev = train_test_split(X_rev, y, test_size=0.33, random_state=50)
print("Regular data is ready")


# For twitter tokenized texts
token = TweetTokenizer()
twitter_summaries = np.array([token.tokenize(line) for line in summaries])
twitter_reviews = np.array([token.tokenize(line) for line in reviews])

summaries_mat_twitter = [one_hot(" ".join(line), vocab_size) for line in twitter_summaries]
reviews_mat_twitter = [one_hot(" ".join(line), vocab_size) for line in twitter_reviews]

X_sum_twitter = pad_sequences(summaries_mat_twitter, maxlen=max_sentence, padding='pre')
X_rev_twitter = pad_sequences(reviews_mat_twitter, maxlen=max_sentence, padding='pre')


X_sum_train_twitter, X_sum_dev_twitter, y_sum_train_twitter, y_sum_dev_twitter = train_test_split(X_sum_twitter, y, test_size=0.33, random_state=50)
X_rev_train_twitter, X_rev_dev_twitter, y_rev_train_twitter, y_rev_dev_twitter = train_test_split(X_rev_twitter, y, test_size=0.33, random_state=50)
print("Regular data is ready")


vocab_size = 10000
embedding_dim = 50
max_sentence = 500
drop_out = 0.25


def create_model(model_name, conv_layer, num_filter, lstm_layer, lstm_neuron, optimizer="adam",learn_rate=0.0001, momentum=0.0, init_mode = "uniform", weight_constraint = 1, dropout_rate = 0.25, verbose = 0):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sentence))
    model.add(Dropout(dropout_rate))
    
    for _ in range(conv_layer):
        model.add(Conv1D(filters=num_filter, kernel_size=3))
        model.add(Activation("relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
    
    if model_name[0] == "C":
        model.add(Flatten())
        for i in range(lstm_layer):
            model.add(Dense(lstm_neuron))
            model.add(Dropout(dropout_rate))
            model.add(Activation("relu"))
    else:
        for i in range(lstm_layer):
            if lstm_layer > 1 and i != lstm_layer - 1:
                r_sequences = True
            else:
                r_sequences = False
            model.add(LSTM(lstm_neuron,dropout=dropout_rate,recurrent_dropout=dropout_rate,return_sequences=r_sequences))
            model.add(Activation("relu"))
    
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if verbose:
        print(model.summary())

    return model
    
    


# The final models
models = {
    'CNN2': {'model_name':"CNN2", 'conv_layer':1, 'num_filter':40, 'lstm_layer':1, 'lstm_neuron':100},
    'LSTM14': {'model_name':"LSTM14", 'conv_layer':1, 'num_filter':80, 'lstm_layer':1, 'lstm_neuron':64}
}

for params in models.values():
    
    print()
    print(params['model_name'])
    model = create_model(**params)

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min')
    
    print("{} with regular data".format(params['model_name']))
    model.fit(X_rev_train, 
              y_rev_train, 
              epochs=10, 
              batch_size=500, 
              validation_data=(X_rev_dev,y_rev_dev),
              callbacks=[early_stopping,reduce_lr])

    print("{} with twitter tokenized data".format(params['model_name']))
    
    model = create_model(**params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min')
    
    model.fit(X_rev_train_twitter, 
              y_rev_train_twitter, 
              epochs=10, 
              batch_size=500, 
              validation_data=(X_rev_dev_twitter,y_rev_dev_twitter),
              callbacks=[early_stopping,reduce_lr]
             )