import numpy as np
import pandas as pd
import lightgbm as lgbm
import os
from utils import *
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from bayes_opt import BayesianOptimization

class ModelSelection:
    def __init__(self,filename,target_col,model,extract_features,k_splits,version,seed):
        self.filename = filename
        self.target_col = target_col
        self.model = model
        self.extract_features = extract_features
        self.k_splits = k_splits
        self.version = version
        self.seed = seed

        self.fitted = False
        self.trains = list()
        self.devs = list()

        if not os.path.isdir("parameters/"):
            os.mkdir("parameters/")

    def bayesian_optimize(self,df,parameter_bounds,parameter_types,n_iters=10):
        self.X,y = df.drop(self.target_col,axis=1),df[self.target_col]
        self.y,_ = pd.factorize(y)
        self.parameter_types = parameter_types

        optimizer = BayesianOptimization(
            f=self._kfold,
            pbounds=parameter_bounds,
            random_state=self.seed,
            verbose=3
        )
        optimizer.maximize(
            n_iter=n_iters,
            init_points=4
        )

        results = pd.DataFrame().from_dict(optimizer.res)
        results["params"] = results["params"].apply(lambda x: self._fix(x))
        results.to_pickle("parameters/{}_{}.pkl".format(self.version,type(self.model()).__name__))
        return results

    def _kfold(self,**kwargs):
        params = self._fix(kwargs)

        folds = KFold(n_splits=self.k_splits,shuffle=True,random_state=self.seed)
        dev_lls = list()

        for k,(idx_train,idx_dev) in enumerate(folds.split(self.X,self.y)):
            if self.fitted:
                X_train,y_train = self.trains[k]
                X_dev,y_dev = self.devs[k]
            else:
                X_train,y_train = self.X.iloc[idx_train],self.y[idx_train]
                X_dev,y_dev = self.X.iloc[idx_dev],self.y[idx_dev]

                # Extract features and pick the most important ones.
                (X_train,feats),X_dev = self.extract_features(X_train,X_dev)
                #imp_feats = get_most_important_features()
                #feat_cols = feats.loc[feats["feature"].isin(imp_feats["feature"])].index
                #X_train = X_train[:,feat_cols]
                #X_dev = X_dev[:,feat_cols]
                self.trains.append((X_train,y_train))
                self.devs.append((X_dev,y_dev))

            clf = self.model(**params)
            clf.fit(
                X_train,
                y_train,
                eval_set = [(X_train,y_train),(X_dev,y_dev)],
                eval_metric = ["binary_logloss","binary_error"],
                early_stopping_rounds = 150,
                verbose=0
            )
            dev_pred_proba = clf.predict_proba(X_dev)
            dev_lls.append(log_loss(y_dev,dev_pred_proba))

            del clf
        self.fitted = True
        
        # Since we're maximizing, we maximize the negative loglikehood
        return -np.mean(dev_lls)
    
    def _fix(self,bounds):
        params = dict()
        for k,v in bounds.items():
            if self.parameter_types[k] == int:
                params[k] = int(np.round(v))
        return params

