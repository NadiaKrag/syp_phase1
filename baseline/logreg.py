import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

class LogisticRegression():
    
    def __init__(self, output_dim, activation, optimizer, loss, metrics, epochs, batch_size):
        self.output_dim = output_dim
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self,X,y):
        self.class_labels = np.unique(y)
        self.model = Sequential()
        self.model.add(Dense(self.output_dim,input_dim=X.shape[1],activation=self.activation))
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
        self.model.fit(X,y,epochs=self.epochs,batch_size=self.batch_size)
        return self.model
        
    def predict(self,X):
        preds = self.model.predict(X)
        return np.where(preds < 0.5, *self.class_labels)

if __name__ == "__main__":
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn import preprocessing
    from scipy.sparse import coo_matrix, hstack, csr_matrix

    data = pd.read_csv('../data/phase1_movie_reviews-train.csv')

    summaries = data["summary"].astype(str).values
    reviews = data["reviewText"].astype(str).values
    year = pd.get_dummies(data["year"])

    vectorizer1 = CountVectorizer()
    vectorizer2 = CountVectorizer(max_features=30000)

    summaries_vec = vectorizer1.fit_transform(summaries)
    reviews_vec = vectorizer2.fit_transform(reviews)

    X = csr_matrix(hstack([summaries_vec,reviews_vec,year]))
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(data["polarity"])

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.33, random_state=50)

    clf = LogisticRegression(output_dim=1,activation="sigmoid",optimizer=optimizers.adam(lr=0.1),\
                         loss="binary_crossentropy",metrics=["accuracy"],epochs=20,batch_size=100)
    clf.fit(X_train,y_train)
    y_hat = clf.predict(X_dev)
    accuracy = accuracy_score(y_dev,y_hat)
    print(accuracy)

