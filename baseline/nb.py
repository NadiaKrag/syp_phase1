import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score

class NB:
    def __init__(self, smoothing = 0.1, vocab = None):
        '''
        Initialize parameters

        :param smoothing: float, the smoothing constant
        :param voacb: list of words, the models vocabulary. If None, vocab will be created in fit.

        '''
        self.fitted = False
        self.vocab = vocab
        self.smoothing = smoothing

        

    def estimate_pxy(self, X, y, label):
        '''
        Compute smoothed log-probability P(word | label) for a given label.

        :param x: list of counts, one per instance
        :param y: list of labels, one per instance
        :param label: desired label
        :returns: defaultdict containing log probabilities per word
        :rtype: defaultdict

        '''

        _X = [X[i] for i in range(len(y)) if y[i] == label]

        log_prob = defaultdict(int)

        for word in self.vocab:
            log_prob[word] = self.smoothing

        for bow in _X:
            for key, count in bow.items():
                if key in self.vocab:
                    log_prob[key] += count

        denumoritor = sum(log_prob.values())

        for word, val in log_prob.items():
            log_prob[word] = np.log(val/denumoritor)
        return log_prob


    def fit(self, X, y):
        """Fits the naive bayes model. Uses BOW as input and return nothing.

        :param X: list of counts, one for each instance
        :param y: list of labels, one for each instance
        :returns: nothing
        """
        assert len(X) == len(y)

        # Initialize self.labels, self.p and self.vocab if none is given
        self.labels, freq = np.unique(y, return_counts=True)
        self.p = np.log(freq/len(y))

        if self.vocab is None:
            words = set()
            for bow in X:
                words = words.union(set(bow.keys()))
            self.vocab = list(words)


        prob_dicts = [self.estimate_pxy(X,y,label) for label in self.labels]

        feature_dict = defaultdict(int)
        for i,label in enumerate(self.labels):
            for word,prob in prob_dicts[i].items():
                feature_dict[(label,word)] = prob 

        self.feature_prob = feature_dict
        self.fitted = True
    
    
    def _predict(self, X):
        """Predicts a single X value

        :param X: A single BOW
        :returns: predicted label
        :rtype: string
        """
        scores = {self.labels[i]: self.p[i] for i in range(len(self.labels)) }

        for label in self.labels:
            for word, count in X.items():
                scores[label] += self.feature_prob[(label,word)] * count
        
        return max(scores, key=scores.get)

    def predict(self, X):
        """Predicts a list X value

        :param X: list of bows (counters), one for each instance
        :returns: list of the predicted labels
        :rtype: list, containing strings
        """

        assert self.fitted
        y_hat = [self._predict(x_i) for x_i in X]
        return y_hat


    def fit_predict(self, X_train, y_train, X_dev):
        """Fits and predicts a list X value

        :param X_train: list of bows (counters), one for each instance training label
        :param y_train: list of bows labels, one for each instance of training bows
        :param X_dev: list of bows labels, one for each instance to predict
        :returns: list of the predicted labels
        :rtype: list, containing strings
        """
        self.fit(X_train, y_train)
        return self.predict(X_dev)


    def find_best_smoother(self,X_train,y_train,X_dev,y_dev,smoothers, verbose = 0):
        '''Finds the highest scoring smoothing value based on accuracy on the dev data
        After it fit the model with the best smoother

        :param X_train: list of training instances
        :param y_train: list of training labels
        :param X_dev: list of dev instances
        :param y_dev: list of dev labels
        :param smoothers: list of smoothing values
        :param verbose: either 0 for no verbose and 1 for verbose
        :returns: best smoothing value and scores in a dict{smoother: score}
        :rtype: float, dict
        '''

        s_scores = {}
        for smoothing in smoothers:
            self.smoothing = smoothing
            self.fit(X_train,y_train)
            y_hat = self.predict(X_dev)
            s_scores[smoothing] = accuracy_score(y_dev,y_hat)
            if verbose == 1:
                print("Smothing: {0:<10.4f} scored {1:>8.4f}".format(smoothing,s_scores[smoothing]))
        self.smoothing = max(s_scores, key=s_scores.get)
        self.fit(X_train,y_train)
        return max(s_scores, key=s_scores.get), s_scores

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    import pandas as pd

    
    def create_bow(text):
        _text = str(text).split()
        counter = Counter(_text)
        return counter

    data = pd.read_csv('../data/phase1_movie_reviews-train.csv')
    X_train = [create_bow(string) for string in data.reviewText]
    NB = NB()
    y_pred = NB.fit_predict(X_train, data.polarity, X_train)
    print(accuracy_score(list(data.polarity),y_pred))



