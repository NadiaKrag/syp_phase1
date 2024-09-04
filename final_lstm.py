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
from keras.regularizers import L1L2



data = pd.read_csv('../../data/phase1_movie_reviews-train.csv')

year = pd.get_dummies(data["year"])
summaries = data["summary"].astype(str).values
reviews = data["reviewText"].astype(str).values

vocab_size = 10000
max_sentence = 500


lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(data["polarity"])
y = np.array([yi[0] for yi in y])


# For twitter tokenized texts
token = TweetTokenizer()
twitter_reviews = np.array([token.tokenize(line) for line in reviews])

reviews_mat_twitter = [one_hot(" ".join(line), vocab_size) for line in twitter_reviews]

X_rev_twitter = pad_sequences(reviews_mat_twitter, maxlen=max_sentence, padding='pre')


X_rev_train_twitter, X_rev_dev_twitter, y_rev_train_twitter, y_rev_dev_twitter = train_test_split(X_rev_twitter, y, test_size=0.33, random_state=50)


vocab_size = 10000
embedding_dim = 50
max_sentence = 500


def create_model(model_name, conv_layer, num_filter, lstm_layer, lstm_neuron, optimizer="Adam",activation = 'relu',learn_rate=0.0001, momentum=0.0, init_mode = "uniform", weight_constraint = 1, dropout_rate = 0.25, verbose = 0,cnn_para = (3,2), bias_regularizer = None):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sentence))
    model.add(Dropout(dropout_rate))
    
    for _ in range(conv_layer):
        model.add(Conv1D(filters=num_filter, kernel_size = cnn_para[0]))
        model.add(Activation(activation))
        model.add(MaxPooling1D(pool_size=cnn_para[1]))
        model.add(Dropout(dropout_rate))
    
    if model_name[0] == "C":
        model.add(Flatten())
        for i in range(lstm_layer):
            model.add(Dense(lstm_neuron))
            model.add(Dropout(dropout_rate))
            model.add(Activation(activation))
    else:
        for i in range(lstm_layer):
            if lstm_layer > 1 and i != lstm_layer - 1:
                r_sequences = True
            else:
                r_sequences = False
            if bias_regularizer[1]:
                model.add(LSTM(lstm_neuron,dropout=dropout_rate,recurrent_dropout=dropout_rate,return_sequences=r_sequences, bias_regularizer = bias_regularizer[1]))
            else:
                model.add(LSTM(lstm_neuron,dropout=dropout_rate,recurrent_dropout=dropout_rate,return_sequences=r_sequences))
            model.add(Activation(activation))
    
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if verbose:
        print(model.summary())

    return model
    
    
    


# The final models
models = {'LSTM14': {
    'model_name':"LSTM"
    , 'conv_layer':1
    , 'num_filter':80
    , 'lstm_layer':1
    , 'lstm_neuron':64
    ,'optimizer' : 'Adam'
    ,'activation' : 'tanh'
    ,'init_mode' : 'normal'
    ,'dropout_rate' : 0.2
    ,'weight_constraint' : 1
    ,'cnn_para': (3,2)
    ,'bias_regularizer':(None,None)
    }
}

# The parameters
hyp_params = dict(cnn_para = [[2,1],[3,2],[4,2],[5,2],[3,3],[4,3],[4,4],[5,4],[5,5]],
                    bias_regularizer=[(0,L1L2(l1=0.0, l2=0.0)), (1,L1L2(l1=0.01, l2=0.0)),(2, L1L2(l1=0.0, l2=0.01)), (3,L1L2(l1=0.01, l2=0.01))]
                 )

# Search loop
for model_param in models.values():
    print(model_param['model_name'])
    for new_param, values in hyp_params.items():
        best_score = 0
        best_param = None
        print(new_param)
        print(values)
        for param in values:
            model_param[new_param] = param
            t_name = "{}-{}-{}".format(model_param['cnn_para'],model_param['bias_regularizer'][0],time())
            print(t_name)
            tensorboard = TensorBoard(log_dir="param_logs/{}".format(t_name))
            

            model = create_model(**model_param)
            

            
            model.fit(X_rev_train_twitter, 
                      y_rev_train_twitter, 
                      epochs=5, 
                      validation_data=(X_rev_dev_twitter,y_rev_dev_twitter),
                      callbacks=[tensorboard],
                     verbose = 0)

            model_score = model.evaluate(X_rev_dev_twitter,y_rev_dev_twitter,verbose = 0)[1]
            print("{} \t {}".format(param,model_score))

            if model_score > best_score:
                best_score = model_score
                best_param = param

        model_param[new_param] = best_param
        print("Best {} was {} with an accuracy of {}".format(new_param,best_param,best_score))
        final_model_params = model_param

epochs_batch = [(3,50),(4,50),(5,50),(3,100),(4,100),(5,100),(3,250),(4,250),(5,250)]
for eb in epochs_batch:
    model = create_model(**final_model_params)

    model.fit(X_rev_train_twitter, 
              y_rev_train_twitter, 
              epochs=eb[0], 
              validation_data=(X_rev_dev_twitter,y_rev_dev_twitter),
              callbacks=[tensorboard],
             batch_size= = eb[1])

    model_score = model.evaluate(X_rev_dev_twitter,y_rev_dev_twitter,verbose = 0)[1]
    print("{} \t {}".format(eb,model_score))
print("DONE!")