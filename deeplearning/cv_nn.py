# from nn_model import NNModel
from sklearn import preprocessing
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

vocab_size = 30000
max_length = 500

data = pd.read_csv('../../data/phase1_movie_reviews-train.csv')
test_movie = pd.read_csv('../../data/phase1_movie_reviews-test-hidden.csv')
test_video = pd.read_csv('../../data/phase1_video_games-test-hidden.csv')
X = data["reviewText"].astype(str).values
X_test_movie = test_movie["reviewText"].astype(str).values
X_test_video = test_video["reviewText"].astype(str).values

lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(data["polarity"])
y = np.array([yi[0] for yi in y])
print(data["polarity"])
print(y)
exit()

kf = KFold(n_splits=7, random_state=7, shuffle=True)


oof_pred = np.zeros(y.shape)
test_preds = []
test_movie_preds = []
test_video_preds = []
for train, dev in kf.split(X):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X[train])
    word_index = tokenizer.word_index

    X_train = tokenizer.texts_to_sequences(X[train])
    X_dev = tokenizer.texts_to_sequences(X[dev])
    X_test_movie = tokenizer.texts_to_sequences(X_test_movie)
    X_test_movie = pad_sequences(X_test_movie, maxlen=max_length)

    X_test_video = tokenizer.texts_to_sequences(X_test_video)
    X_test_video = pad_sequences(X_test_video, maxlen=max_length)

    X_train = pad_sequences(X_train, maxlen=max_length)
    X_dev = pad_sequences(X_dev, maxlen=max_length)

    model = NNModel(model_fname="cnn2.keras",emb_fname='models/gl_w2v_org.emb', vocab_size=vocab_size)
    model.fit(X_train, y[train], word_index, X_dev, y[dev])
    oof_pred[dev] = 1 - model.predict(X_dev)
    test_movie_preds.append(1 - model.predict(X_test_movie))
    test_video_preds.append(1 - model.predict(X_test_video))
oof_pred.savetxt('results/cnn2_review_phase1_movie_reviews-train.csv')
np.mean(test_movie_preds).savetxt('results/cnn2_review_phase1_movie_reviews-test-hidden.csv')
np.mean(test_video_preds).savetxt('results/cnn2_review_phase1_video_games-test-hidden.csv')


