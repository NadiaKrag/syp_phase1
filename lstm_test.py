import re
import numpy as np
import pandas as pd
from time import time


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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




data = pd.read_csv('../data/phase1_movie_reviews-train.csv')

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


# CNN and CNN Dense
conv_layers = [1,2,3]
num_filters = [40, 80, 140]
dense_layers = [0,1, 2]
hidden_neurons = [50, 100]
drop_out = 0.25


model_id = 1

for conv_layer in conv_layers:
    for num_filter in num_filters:
        for dense_layer in dense_layers:
            for hidden_neuron in hidden_neurons:
                if dense_layer == 0 and hidden_neuron > 32:
                    continue
                
                model_name = "CNN{}--conv{}-filters{}-lstm{}-lstmN{}-time{}".format(model_id,conv_layer,num_filter,dense_layer,hidden_neurons,int(time()))
                        
                model = Sequential()

                model.add(Embedding(vocab_size, 50, input_length=max_sentence))
                model.add(Dropout(drop_out))
                

                for _ in range(conv_layer):
                    model.add(Conv1D(filters=num_filter, kernel_size=3))
                    model.add(Activation("relu"))
                    model.add(MaxPooling1D(pool_size=2))
                    model.add(Dropout(drop_out))
                    
                model.add(Flatten())
                
                for i in range(dense_layer):
                    model.add(Dense(hidden_neuron))
                    model.add(Dropout(drop_out))
                    model.add(Activation("relu"))

                
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['accuracy'])
                print(model.summary())
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min')
                tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

                model.fit(X_rev_train, y_rev_train, epochs=10, batch_size=500, validation_data=(X_rev_dev,y_rev_dev),callbacks=[tensorboard,early_stopping,reduce_lr])
                model_id += 1





                
# LSTM and CNN LSTM
conv_layers = [0,1]
num_filters = [40, 80, 140]
lstm_layers = [1, 2]
lstm_neurons = [32, 64, 128]
drop_out = 0.25


model_id = 1

for conv_layer in conv_layers:
    for num_filter in num_filters:
        for lstm_layer in lstm_layers:
            for lstm_neuron in lstm_neurons:
                if conv_layer == 0 and num_filter > 40:
                    continue
                
                model_name = "lstm{}--conv{}-filters{}-lstm{}-lstmN{}-time{}".format(model_id,conv_layer,num_filter,lstm_layer,lstm_neuron,int(time()))
                        
                model = Sequential()

                model.add(Embedding(vocab_size, 50, input_length=max_sentence))
                model.add(Dropout(drop_out))
                

                for _ in range(conv_layer):
                    model.add(Conv1D(filters=num_filter, kernel_size=3))
                    model.add(Activation("relu"))
                    model.add(MaxPooling1D(pool_size=2))
                    model.add(Dropout(drop_out))

                for i in range(lstm_layer):
                    if lstm_layer > 1 and i != lstm_layer - 1:
                        r_sequences = True
                    else:
                        r_sequences = False
                    model.add(LSTM(lstm_neuron,dropout=drop_out,recurrent_dropout=drop_out,return_sequences=r_sequences))
                    model.add(Activation("relu"))

                
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['accuracy'])
                print(model.summary())
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min')
                tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

                model.fit(X_rev_train, y_rev_train, epochs=10, batch_size=500, validation_data=(X_rev_dev,y_rev_dev),callbacks=[tensorboard,early_stopping,reduce_lr])
                model_id += 1





