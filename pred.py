import numpy as np
import pandas as pd
import lightgbm as lgbm
from keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
# Takes care of annoying sklearn warnings.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

train_filename = "phase1_movie_reviews-train.csv"
test1_filename = "phase1_movie_reviews-test-hidden.csv"
test2_filename = "phase1_video_games-test-hidden.csv"
target_col = "polarity"
k_splits = 7

### Load data.
df_train = pd.read_csv("../data/raw/{}".format(train_filename))
X_test1 = pd.read_csv("../data/raw/{}".format(test1_filename)).drop(target_col,axis=1)
X_test2 = pd.read_csv("../data/raw/{}".format(test2_filename)).drop(target_col,axis=1)
X,(y,classes) = df_train.drop(target_col,axis=1),pd.factorize(df_train[target_col])

### Preprocessing.
def negate_sentences(data,col):
    docs = [text_to_word_sequence(string,filters='',lower=False) for string in data[col].astype(str).values]
    final_result = []
    for doc in docs:
        negation = False
        delims = "?.,!:;"
        result = []
        for token in doc:
            if token in delims:
                negation = False
            stripped = token.lower()
            negated = "NOT_" + stripped if negation else token
            result.append(negated)
            if stripped.strip(delims) in ["not","no","never"] or stripped.strip(delims)[-3:] == "n't":
                negation = True
            elif token[-1] in delims:
                negation = False
        final_result.append(" ".join(result))
    data[col] = final_result
    return data

def deal_with_nans(df):
    return df.fillna("")

def preprocess(df,col,neg=False):
    if neg:
        df = negate_sentences(df,col)
    return deal_with_nans(df)[col]

column = "summary"
X = preprocess(X,column,neg=False)
X_test1 = preprocess(X_test1,column,neg=False)
X_test2 = preprocess(X_test2,column,neg=False)

### Predictions.
folds = KFold(k_splits,shuffle=True,random_state=7)
dev_pred_probas = np.zeros(y.shape)
test1_pred_probas = np.zeros(X_test1.shape[0])
test2_pred_probas = np.zeros(X_test2.shape[0])

for idx,(train_idx,dev_idx) in enumerate(folds.split(X,y)):
    print("fold number {}".format(idx+1))
    vect = TfidfVectorizer(
        tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True).tokenize,
        ngram_range=(1,2),
        analyzer="word",
        max_df=0.10698949287056575,
        min_df=1,
        max_features=None,
        binary=False,
        stop_words=None,
        dtype=np.float64,
        norm = "l2",
        use_idf = True,
        smooth_idf = True,
        sublinear_tf = False
    )
    X_train,y_train = X.iloc[train_idx],y[train_idx]
    X_dev,y_dev = X.iloc[dev_idx],y[dev_idx]
    
    X_train = vect.fit_transform(X_train)
    X_dev = vect.transform(X_dev)
    X_test1_cur = vect.transform(X_test1)
    X_test2_cur = vect.transform(X_test2)

    clf = lgbm.LGBMClassifier(
        colsample_bytree = 0.3662,
        learning_rate = 0.1,
        max_depth = 8,
        min_child_samples = 5,
        min_child_weight = 0.04231,
        min_split_gain = 0.7798,
        n_estimators = 1976,
        num_leaves = 58,
        reg_alpha = 0.2316,
        reg_lambda = 0.941
    )
    clf.fit(
        X_train,
        y_train,
        eval_set = [(X_train,y_train),(X_dev,y_dev)],
        eval_metric = ["binary_logloss","binary_error"],
        early_stopping_rounds = 150,
        verbose=0
    )
    dev_pred_probas[dev_idx] = clf.predict_proba(X_dev)[:,0]
    test1_pred_probas = clf.predict_proba(X_test1_cur)[:,0] / k_splits
    test2_pred_probas = clf.predict_proba(X_test2_cur)[:,0] / k_splits

    # To see how it's going
    oof_pred = np.where(dev_pred_probas[dev_idx] < 0.5,1,0)
    print("oof acc:",accuracy_score(y_dev,oof_pred),"\n")


model_name = type(clf).__name__
pd.DataFrame(dev_pred_probas).to_csv("predictions/{}_{}_{}.csv".format(model_name,column,train_filename.split(".")[0]),index=False)
pd.DataFrame(test1_pred_probas).to_csv("predictions/{}_{}_{}.csv".format(model_name,column,test1_filename.split(".")[0]),index=False)
pd.DataFrame(test2_pred_probas).to_csv("predictions/{}_{}_{}.csv".format(model_name,column,test2_filename.split(".")[0]),index=False)



