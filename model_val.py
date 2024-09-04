import numpy as np
import pandas as pd
import lightgbm as lgbm
import os
from utils import *
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,log_loss

class ModelValidation:
    def __init__(self,target_col,model,extract_features,k_splits,version,seed):
        self.target_col = target_col
        self.model = model
        self.extract_features = extract_features
        self.k_splits = k_splits
        self.version = version
        self.seed = seed

        if not os.path.isdir("validation/"):
            os.mkdir("validation/")
        
    def validate(self,df,n_repeats=3,save=False):
        X,y = df.drop(self.target_col,axis=1),df[self.target_col]
        y,_ = pd.factorize(y)
        accs = [0]*self.k_splits
        lls = [0]*self.k_splits

        self.pbar = tqdm(total=n_repeats*self.k_splits) # Loading bar.
        for n in range(n_repeats):
            acc,ll = self._kfold(X,y,n)
            accs = [sum(x) for x in zip(acc,accs)]
            lls = [sum(x) for x in zip(ll,lls)]
        self.pbar.close()
        
        accs = [acc/n_repeats for acc in accs]
        lls = [ll/n_repeats for ll in lls]

        print("avg oof accuracy: {:>7.4f} ({:.4f})".format(np.mean(accs),np.std(accs)))
        print("avg oof logloss: {:>8.4f} ({:.4f})".format(np.mean(lls),np.std(lls)))

        if save:
            self._save_results(accs,lls)
    
    def _kfold(self,X,y,n):
        folds = KFold(n_splits=self.k_splits,shuffle=True,random_state=n*self.seed)
        # dev_preds = np.zeros(X.shape[0])
        # dev_pred_probas = np.zeros(X.shape[0])
        dev_accs = list()
        dev_lls = list()

        for (idx_train,idx_dev) in folds.split(X,y):
            X_train,y_train = X.iloc[idx_train],y[idx_train]
            X_dev,y_dev = X.iloc[idx_dev],y[idx_dev]

            # Extract features and pick the most important ones.
            (X_train,feats),X_dev = self.extract_features(X_train,X_dev)
            imp_feats = get_most_important_features()
            feat_cols = feats.loc[feats["feature"].isin(imp_feats["feature"])].index
            X_train = X_train[:,feat_cols]
            X_dev = X_dev[:,feat_cols]

            clf = self.model()
            clf.fit(X_train,y_train)
            dev_pred = clf.predict(X_dev)
            dev_pred_proba = clf.predict_proba(X_dev)
            # dev_preds[idx_dev] = dev_pred
            # dev_pred_probas[idx_dev] = dev_pred_proba
            dev_accs.append(accuracy_score(y_dev,dev_pred))
            dev_lls.append(log_loss(y_dev,dev_pred_proba))

            self.pbar.update(1)
            del clf
        
        return dev_accs,dev_lls

    def _save_results(self,accs,lls):
        results = pd.DataFrame(
            columns = ["accuracy","logloss"]
        )
        results["accuracy"] = accs
        results["logloss"] = lls
        results.to_pickle("validation/{}_{}.pkl".format(self.version,type(self.model()).__name__))


