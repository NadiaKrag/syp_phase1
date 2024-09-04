import numpy as np
import pandas as pd
import lightgbm as lgbm
import os
from tqdm import tqdm

class FeatureSelection:
    def __init__(self,target_col,version,seed):
        self.target_col = target_col
        self.version = version
        self.seed = seed

        if not os.path.isdir("features/"):
            os.mkdir("features/")

    def null_selection(self,X,y):
        # Find the feature importances without shuffling the target.
        actual_feat_imps = self._lgbm(X,y)
        
        # Find the distribution of feature importances after shuffling the target.
        null_feat_imps = pd.DataFrame()
        pbar = tqdm(total=100) # Loading bar.
        for i in range(100):
            np.random.seed(i*self.seed)
            y_null = np.random.permutation(y)
            null_feat_imp = self._lgbm(X,y_null)
            # feat_imps = pd.DataFrame(
            #     data=np.column_stack([X.columns,feat_imps_gain]),
            #     columns=["feature","gain"]
            # )
            null_feat_imps = pd.concat([null_feat_imps,null_feat_imp],axis=0)
            pbar.update(1)
        pbar.close()
        
        actual_feat_imps.to_pickle("features/{}_actual.pkl".format(self.version))
        null_feat_imps.to_pickle("features/{}_null.pkl".format(self.version))

        return actual_feat_imps,null_feat_imps

    def _lgbm(self,X,y):
        clf = lgbm.train(
            {
                "objective":"binary",
                "num_threads":-1,
                "verbose":-1
            },
            lgbm.Dataset(X,y),
            num_boost_round=1500
        )
        return pd.DataFrame(clf.feature_importance(importance_type="gain"),columns=["gain"])