import numpy as np
from collections import Counter

class ClassDist:
    def __init__(self):
        self.fitted = False
        self.labels = None
        self.p = None

    def fit(self, X, y):
        self.labels = sorted(list(set(y)))
        count = Counter(y)
        self.p = [count[label]/len(y) for label in self.labels]
        self.fitted = True

    def predict(self, X):
        assert self.fitted
        return np.random.choice(self.labels,
                                (len(X)),
                                replace=True,
                                p=self.p)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    import pandas as pd

    data = pd.read_csv('../data/phase1_movie_reviews-train.csv')

    clf = ClassDist()
    y_pred = clf.fit_predict(data.reviewText, data.polarity)
    print(accuracy_score(list(data.polarity),y_pred))


