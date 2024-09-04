import numpy as np
import pandas as pd
import pickle
import os
from keras.preprocessing.text import text_to_word_sequence

class Preprocessing:
    def __init__(self,filename,version,input_dir="../data/raw/",output_dir="../data/preprocessed/"):
        self.filename = filename
        self.version = version
        self.input_dir = input_dir
        self.output_dir = output_dir
    
    def read_raw(self):
        return pd.read_csv(self.input_dir + self.filename +".csv")

    def lowercase(self,df,cols):
        return pd.concat([df[col].str.lower() for col in cols],axis=1)

    def deal_with_nans(self,df):
        return df.fillna("")

    def negate_sentences(self,data,col):
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
    
    def remove_quots(self,data,col):
        result = []
        for index, row in data.iterrows():
            if '&quot;' in str(row[col]):
                x = re.sub('&quot;', '', str(row[col]))
                result.append(x)
            else:
                result.append(row[col])
        data[col] = result
        return data

    def save(self,df):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        new_filename = self.output_dir + self.version + "_" + self.filename + ".pkl"
        df.to_pickle(new_filename)

if __name__ == "__main__":

    data = pd.read_csv('../data/raw/phase1_movie_reviews-train.csv')

    obj = Preprocessing("fdfd","Efdfs")
    data = obj.negate_sentences(data,"summary")

    print(data)



