import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score,log_loss,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold

train_filename = "phase1_movie_reviews-train.csv"
test1_filename = "phase1_movie_reviews-test-hidden.csv"
test2_filename = "phase1_video_games-test-hidden.csv"
target_col = "polarity"
k_splits = 7

y_ = pd.read_csv("../data/raw/{}".format(train_filename))["polarity"]
y,classes = pd.factorize(y_)

### Get predictions.
X = pd.DataFrame()
X_test1 = pd.DataFrame()
X_test2 = pd.DataFrame()
for pred_proba in os.listdir("predictions/"):
    model,col,filename = pred_proba.split("_",maxsplit=2)
    if filename == train_filename:
        X[model + "_" + col] = pd.read_csv("predictions/{}".format(pred_proba)).values.reshape((-1,))
    elif filename == test1_filename:
        X_test1[model + "_" + col] = pd.read_csv("predictions/{}".format(pred_proba)).values.reshape((-1,))
    elif filename == test2_filename:
        X_test2[model + "_" + col] = pd.read_csv("predictions/{}".format(pred_proba)).values.reshape((-1,))

X.columns = sorted(X.columns)
X_test1.columns = sorted(X_test1.columns)
X_test2.columns = sorted(X_test2.columns)
logreg = True

if logreg:
    folds = KFold(k_splits,shuffle=True,random_state=7)
    dev_pred_probas = np.zeros(y.shape)
    dev_preds = np.zeros(y.shape)

    test1_pred_probas = np.zeros(X_test1.shape[0])
    test2_pred_probas = np.zeros(X_test2.shape[0])

    for idx,(train_idx,dev_idx) in enumerate(folds.split(X,y)):
        print("fold {}".format(idx+1))
        X_train,y_train = X.iloc[train_idx],y[train_idx]
        X_dev,y_dev = X.iloc[dev_idx],y[dev_idx]

        clf = LogisticRegression(
            solver="liblinear"
        )

        clf.fit(X_train,y_train)
        print(X_train.columns)
        print(clf.coef_)
        dev_pred_probas[dev_idx] = clf.predict_proba(X_dev)[:,0]
        dev_preds[dev_idx] = clf.predict(X_dev)

        test1_pred_probas += clf.predict_proba(X_test1)[:,0] / k_splits
        test2_pred_probas += clf.predict_proba(X_test2)[:,0] / k_splits

    acc = accuracy_score(y,dev_preds)
    loss = log_loss(y,1-dev_pred_probas)
    print("oof acc:",acc)
    print("oof logloss:",loss)
    print(classification_report(y,dev_preds))
    
    
    test1_preds = np.where(test1_pred_probas < 0.5,classes[1],classes[0])
    test2_preds = np.where(test2_pred_probas < 0.5,classes[1],classes[0])

    test1 = pd.read_csv("../data/raw/true_movie_labels.txt",header=None)
    test2 = pd.read_csv("../data/raw/true_game_labels.txt",header=None)

    print("movie acc:", accuracy_score(test1,test1_preds))
    print("movie ll:", log_loss(test1,1-test1_pred_probas))

    print("game acc:", accuracy_score(test2,test2_preds))
    print("game ll:", log_loss(test2,1-test2_pred_probas))

    pd.DataFrame(test1_preds).to_csv("final_predictions/group03_movie_preds.txt",index=False)
    print("movie_preds")
    print(np.unique(test1_preds,return_counts=True))
    pd.DataFrame(test2_preds).to_csv("final_predictions/group03_game_preds.txt",index=False)
    print("game_preds")
    print(np.unique(test2_preds,return_counts=True))

    exit()


######################## here we stop

### Predictions
def _kfold(**kwargs):
    folds = KFold(k_splits,shuffle=True,random_state=7)
    dev_pred_probas = np.zeros(y.shape)
    losses = list()

    for idx,(train_idx,dev_idx) in enumerate(folds.split(X,y)):
        X_train,y_train = X.iloc[train_idx],y[train_idx]
        X_dev,y_dev = X.iloc[dev_idx],y[dev_idx]
        
        weight_avg = np.zeros(y_dev.shape)
        for idx,col in enumerate(X_dev.columns):
            key = "weight_{}".format(idx)
            weight_avg += kwargs[key] * X_dev[col]
            
        # one minus because we have probabilities of negative classes (0) and not positive (1)
        losses.append(log_loss(y_dev,1-weight_avg))
    
    return -np.mean(losses)

parameter_bounds = {
    "weight_0": (0,1),
    "weight_1": (0,1),
    "weight_2": (0,1),
    "weight_3": (0,1),
    "weight_4": (0,1)
}
optimizer = BayesianOptimization(
    f = _kfold,
    pbounds = parameter_bounds,
    random_state = 7,
    verbose = 3
)
optimizer.maximize(
    n_iter = 100, # 100 iterations takes about 15 minutes
    init_points = 4
)
results = pd.DataFrame().from_dict(optimizer.res).sort_values(by="target",ascending=False)
best_weights = results.iloc[0]["params"]
print("best weights:", best_weights)

test1_pred_probas = np.zeros(X_test1.shape[0])
test2_pred_probas = np.zeros(X_test2.shape[0])

for idx,col in enumerate(X.columns):
    key = "weight_{}".format(idx)
    test1_pred_probas += best_weights[key] * X_test1[col]
    test1_pred_probas += best_weights[key] * X_test1[col]

if not os.path.isdir("final_predictions/"):
    os.mkdir("final_predictions/")

test1_preds = np.where(test1_pred_probas < 0.5,classes[1],classes[0])
test2_preds = np.where(test2_pred_probas < 0.5,classes[1],classes[0])

pd.Series(test1_preds).to_csv("final_predictions/group03_movie_preds.txt")
pd.Series(test2_preds).to_csv("final_predictions/group03_game_preds.txt")



