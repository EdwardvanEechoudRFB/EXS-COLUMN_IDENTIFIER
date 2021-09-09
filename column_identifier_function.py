# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 17:31:12 2021

@author: Edward van Eechoud
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
df_test = pd.read_csv('c://users//edward van eechoud//downloads//export.csv').tail(1000)
from copy import deepcopy
from sklearn.model_selection import train_test_split
from math import ceil,floor
from ftfy import fix_text
import re # amazing text cleaning for decode issues..

import nltk



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier


from sklearn.base import BaseEstimator, TransformerMixin


def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):

        return X[self.field]
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):

        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

class ec(BaseEstimator):
    def __init__(self,col):
        self.col = col
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, I):
        if isinstance(I,list):
            I = pd.DataFrame({self.col:I})
        X = deepcopy(I)
        X['total_length'] = X[self.col].apply(lambda x: len(x))
        X['length_numbers'] = X[self.col].apply(lambda x: sum(c.isdigit() for c in x))
        X['length_no_spaces'] = X[self.col].apply(lambda x: len(x.replace(' ','')))
        X['length_letters'] = X[self.col].apply(lambda x: sum(c.isalpha() for c in x))
        return X 


def prep_df(idf):
    l = len(idf)
    dfs = np.array_split (idf,floor(l/13))
    output = []
    output_2 = []
    for df in tqdm(dfs):
        df  = df.stack().reset_index()
        df = df.rename(columns = {'level_1':'type',0:'val'})
        df = df.drop('level_0',axis=1)
        output_2.append(df)
        dl = pd.DataFrame(df.groupby(by=['type']).apply(lambda x: x['val'].to_list()))
        dl = dl.reset_index().rename(columns = {0:'val'})
        
        output.append(dl)

    return pd.concat(output),pd.concat(output_2)
        


def ngrams(string, n=3):
    string = str(string)
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.upper() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams] 


print('starting prepping')
f , var2 = prep_df(df_test)
var2.rename(columns={'val':'Text','type':'Label'},inplace=True)


X = var2[['Text']]
Y = var2['Label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)





classifier = Pipeline([
    ('preprocess', ec('Text')),
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector('Text')),
            ('tfidf', TfidfVectorizer(
                     min_df=.0025, max_df=0.25, analyzer=ngrams)),
            ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
        ])),
        ('total_lenght', Pipeline([
            ('wordext', NumberSelector('total_length')),
            ('wscaler', StandardScaler()),
        ])),
        ('length_numbers', Pipeline([
            ('wordext', NumberSelector('length_numbers')),
            ('wscaler', StandardScaler()),
        ])),
        ('length_no_spaces', Pipeline([
            ('wordext', NumberSelector('length_no_spaces')),
            ('wscaler', StandardScaler()),
        ])),
        ('length_letters', Pipeline([
            ('wordext', NumberSelector('length_letters')),
            ('wscaler', StandardScaler()),
        ])),
    ])),
    #('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
    ('clf', RandomForestClassifier()),
    ])



classifier.fit(X_train, y_train)

preds = classifier.predict(X_test)

classifier.predict([''])

accuracy_score(y_test, preds)
classification_report(y_test, preds)
confusion_matrix(y_test, preds)

























