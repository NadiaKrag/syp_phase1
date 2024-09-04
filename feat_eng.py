import numpy as np
import pandas as pd
import re
import os 
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import text_to_word_sequence
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.tokenize import TweetTokenizer

class FeatureEngineering:
    def __init__(self,filename,target_col,version,directory="data/preproc/"):
        self.filename = filename
        self.target_col = target_col
        self.version = version
        self.directory = directory

        if not os.path.isdir("features/"):
            os.mkdir("features/")

    def read_preprocessed(self):
        return pd.read_pickle(self.directory + self.filename + ".pkl").drop(self.target_col,axis=1)

    def reshape_sparse(self,features):
        to_stack = [np.array(feature).reshape(len(feature),1) if type(feature) == list else feature for feature in features]
        print(hstack([csr_matrix(feature) for feature in to_stack]).shape)
        return hstack([csr_matrix(feature) for feature in to_stack])

    def fit(self,df,col="summary"):
        #self.fit_BOW(df,col)
        #self.fit_tfidf(df,col)
        self.fit_NMF(df,col,tfidf=True)
        self.fit_LDA(df,col)

    def transform(self,df,col="summary",save=False,return_features=False):
        # bag-of-words features
        #X_BOW = self.transform_BOW(df,col)
        #X_tfidf = self.transform_tfidf(df,col)
        X_NMF = self.transform_NMF(df,col,tfidf=True)
        print(X_NMF.shape)
        X_LDA = self.transform_LDA(df,col)
        print(X_LDA.shape)

        # stand-alone features
        X_smiley_count = self.smiley_count(df,col)
        X_word_count = self.word_count(df,col)
        X_avg_word_len = self.avg_word_length(df,col)
        X_upper_words = self.upper_words(df,col)
        X_unique_words = self.unique_words(df,col)
        X_char_count = self.char_count(df,col,["\?","\!","\.","\...","\#","\@"])
        print(X_char_count.shape)
        X_quot_count = self.quot_count(df,col)

        X = self.reshape_sparse([
            X_NMF,
            X_LDA,
            X_smiley_count,
            X_word_count,
            X_avg_word_len,
            X_upper_words,
            X_unique_words,
            X_char_count,
            X_quot_count
        ])

        # list of features - order should correspond with what is returned
        features = list()
        #features += self.tfidf_vect.get_feature_names()
        #features += self.char_count_vect.get_feature_names()

        features_df = pd.DataFrame(features,columns=["feature"])

        if save:
            features_df.to_pickle("features/{}_features.pkl".format(self.version))
        if return_features:
            return X,features_df
        else:
            return X

    def fit_BOW(self,df,col,max_features=165311):
        self.count_vect = CountVectorizer(
            tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True).tokenize,
            ngram_range=(1,1),
            analyzer="word",
            max_df=1.0,
            min_df=1,
            max_features=max_features,
            binary=False,
            stop_words=None,
            dtype=np.float64
        )
        self.count_vect.fit(df[col])

    def transform_BOW(self,df,col):
        return self.count_vect.transform(df[col])

    def fit_tfidf(self,df,col,max_features=None):
        self.tfidf_vect = TfidfVectorizer(
            tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True).tokenize,
            ngram_range=(1,2),
            analyzer="word",
            max_df=0.10698949287056575,
            min_df=1,
            max_features=max_features,
            binary=False,
            stop_words=None,
            dtype=np.float64,
            norm = "l2",
            use_idf = True,
            smooth_idf = True,
            sublinear_tf = False
        )
        self.tfidf_vect.fit(df[col])

    def transform_tfidf(self,df,col):
        return self.tfidf_vect.transform(df[col])

    def fit_NMF(self,df,col,max_features=None,tfidf=False,topics=10):
        self.NMF_model = NMF(n_components=topics)
        if tfidf == True:
            self.fit_tfidf(df,col,max_features)
            features = self.transform_tfidf(df,col)
        else:
            self.fit_BOW(df,col,max_features)
            features = self.transform_BOW(df,col)
        self.NMF_model.fit(features)

    def transform_NMF(self,df,col,tfidf=False):
        if tfidf == True:
            features = self.transform_tfidf(df,col)
        else:
            features = self.transform_BOW(df,col)
        return self.NMF_model.transform(features)

    def fit_LDA(self,df,col,max_features=None,topics=10):
        self.LDA_model = LatentDirichletAllocation(n_components=topics)
        self.fit_BOW(df,col,max_features)
        features = self.transform_BOW(df,col)
        self.LDA_model.fit(features)

    def transform_LDA(self,df,col):
        features = self.transform_BOW(df,col)
        return self.LDA_model.transform(features)

    def smiley_count(self,df,col):
        smiley_count = []
        for index, row in df.iterrows():
            smileys = re.findall(r' (?::|;|=)(?:-)?(?:\)|\(|D|P|p|O|o|\*|\/|\\|S|s)',str(row[col]))
            smiley_count.append(len(smileys))
        return smiley_count

    def word_count(self,df,col):
        tokens = [text_to_word_sequence(string) for string in df[col].astype(str).values]
        word_count = [len(doc) for doc in tokens]
        return word_count

    def avg_word_length(self,df,col):
        tokens = [text_to_word_sequence(string) for string in df[col].astype(str).values]
        average_length = []
        for doc in tokens:
            length = [len(w) for w in doc]
            average_length.append(sum(length)/(len(length)+1e-10))
        return average_length

    def upper_words(self,df,col):
        tokens_no_lower = [text_to_word_sequence(string,lower=False) for string in df[col].astype(str).values]
        upper_count = []
        for doc in tokens_no_lower:
            upper = [w for w in doc if w.isupper()]
            upper_count.append(len(upper))
        return upper_count

    def unique_words(self,df,col):
        tokens = [text_to_word_sequence(string) for string in df[col].astype(str).values]
        unique_count = [len(set(doc)) for doc in tokens]
        return unique_count

    def char_count(self,df,col,char_list):
        empty_df = pd.DataFrame()
        for idx,char in enumerate(char_list):
            empty_df[idx] = df[col].str.count(char)
        return empty_df
        
    def quot_count(self,df,col):
        quot_count = []
        for index, row in df.iterrows():
            count = str(row[col]).count('&quot')
            quot_count.append(count)
        return quot_count

if __name__ == "__main__":

    data = pd.read_csv('../data/raw/phase1_movie_reviews-train.csv')
    data.fillna("",inplace=True)
    obj = FeatureEngineering("fdfd","Efdfs","dfds")
    obj.fit(data)
    obj.transform(data)


