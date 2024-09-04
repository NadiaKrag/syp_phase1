import numpy as np
import pandas as pd
from copy import deepcopy
from prep import Preprocessing
from feat_eng import FeatureEngineering
from feat_sel import FeatureSelection
from model_sel import ModelSelection
from model_val import ModelValidation

# Takes care of annoying sklearn warnings.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Pipeline:
    def __init__(self,filename,target_col,model,k_splits,version,seed):
        self.filename = filename.split(".")[0]
        self.target_col = target_col
        self.model = model.__class__
        self.k_splits = k_splits
        self.version = version
        self.seed = seed

    def preprocessing(self,save=False):
        pp = Preprocessing(
            self.filename,
            self.version
        )

        df = pp.read_raw()
        #df = pp.negate_sentences(df,"summary")
        df = pp.deal_with_nans(df)
        
        if save:
            pp.save(df)
        return df
    
    def feature_engineering(self,X_train=None,X_dev=None):
        fe = FeatureEngineering(
            self.target_col,
            self.version
        )

        if type(X_train) == type(None):
            X_train = fe.read_preprocessed()
        fe.fit(X_train)
        
        if type(X_dev) == type(None):
            return fe.transform(X_train,save=True)
        else:
            return fe.transform(X_train,return_features=True),fe.transform(X_dev)

    def feature_selection(self,X,y):
        print("Selecting features..")
        fs = FeatureSelection(
            self.target_col,
            self.version,
            self.seed
        )
        return fs.null_selection(X,y)
    
    def model_selection(self,df,parameter_bounds,parameter_types,n_iters=10):
        print("Optimizing hyperparameters..")
        ms = ModelSelection(
            self.filename,
            self.target_col,
            self.model,
            self.feature_engineering,
            self.k_splits,
            self.version,
            self.seed
        )

        return ms.bayesian_optimize(df,parameter_bounds,parameter_types,n_iters)
    
    def model_validation(self,X,n_repeats,save=False):
        print("Validating..")
        mv = ModelValidation(
            self.target_col,
            self.model,
            self.feature_engineering,
            self.k_splits,
            self.version,
            self.seed
        )

        mv.validate(X,n_repeats,save)


### Metadata.
version = 2
seed = 1618
filename = "phase1_movie_reviews-train.csv"
target_col = "polarity"

### Set up model and validation system.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
import lightgbm as lgbm

clf = lgbm.LGBMClassifier()
k_splits = 7

### Run code.
pipe = Pipeline(filename,target_col,clf,k_splits,version,seed)
df = pipe.preprocessing()
y,_ = pd.factorize(df[target_col])
X = pipe.feature_engineering(df)
actual,null = pipe.feature_selection(X,y)
# The intervals of which parameters are optimized.
parameter_bounds = {
    "learning_rate":(0.1,0.001),
    "num_leaves":(5,70),
    "max_depth":(3,8),
    "n_estimators":(10,2000),
    "min_split_gain":(0.00001,0.99999),
    "min_child_weight":(1e-5,1e-1),
    "min_child_samples":(5,50),
    "colsample_bytree":(0.01,0.99),
    "reg_alpha":(0.01,0.99),
    "reg_lambda":(0.01,0.99)
}
parameter_types = {
    "learning_rate":float,
    "num_leaves":int,
    "max_depth":int,
    "n_estimators":int,
    "min_split_gain":float,
    "min_child_weight":float,
    "min_child_samples":int,
    "colsample_bytree":float,
    "reg_alpha":float,
    "reg_lambda":float
}
#pipe.model_selection(df,parameter_bounds,parameter_types,n_iters=100)
#pipe.model_validation(df,n_repeats=3,save=True)




print("\nREMEMBER TO INCREMENT VERSION NUMBER.")
