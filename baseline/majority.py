import numpy as np
from collections import Counter

class Majority:
    def __init__(self):
        self.fitted = False
        self.label = None

    def fit(self, X, y):
        count = Counter(y)
        self.label = count.most_common(1)[0][0]
        self.fitted = True

    def predict(self, X):
        assert self.fitted
        return np.repeat(self.label, len(X))

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    import pandas as pd

    data = pd.read_csv('../data/phase1_movie_reviews-train.csv')

    clf = Majority()
    y_pred = clf.fit_predict(data.reviewText, data.polarity)
    print(accuracy_score(list(data.polarity),y_pred))


