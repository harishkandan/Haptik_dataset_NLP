# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TreebankWordTokenizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
import pickle
import time

def timeit(method):

    def timed(*args, **kw):

        ts = time.time()

        result = method(*args, **kw)

        te = time.time()

        if 'log_time' in kw:

            name = kw.get('log_name', method.__name__.upper())

            kw['log_time'][name] = int((te - ts) * 1000)

        else:

            print('%r  %2.2f ms' % \

                  (method.__name__, (te - ts) * 1000))

        return result

    return timed


class Haptik:
    
    def pickle_dump(data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)
        
    def pickle_load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)#, encoding = 'latin1')
        
    def __init__(self, path):
        
        self.path = path
        train_path = path + '/train_data.csv'
        test_path = path + '/test_data.csv'
        
        self.train_data = pd.read_csv(train_path, encoding = 'utf-8')
        self.test_data = pd.read_csv(test_path, encoding = 'utf-8')
        
        self.train_data = self.train_data.sample(frac = 1)
        self.test_data = self.test_data.sample(frac = 1)
        
        self.train_data = self.train_data[self.train_data.apply(lambda x: sum(x == 'T'), axis = 1) == 1]
        
        self.X_train = self.train_data.iloc[:, 0]
        self.X_test = self.test_data.iloc[:, 0]
        self.y_train = self.train_data.iloc[:, 1:]
        self.y_test = self.test_data.iloc[:, 1:]
        self = self.reverse_OHE().preprocess()
        
    def reverse_OHE(self):
        '''
        Parameters
        ------------
        Instance of 'Haptik' object
        
        Function
        ------------
        Contains the 'T's to 1 and 'F's to 0 in X_train and X_test. The resultant DataFrame which would be
        in the form of One Hot Encoding is transformed to avoid curse of dimensionality.
        '''
        for colname in list(self.y_train.columns):
            self.y_train[colname] = self.y_train[colname].astype(str).map({'F':0, 'T':1})
        for colname in list(self.y_test.columns):
            self.y_test[colname] = self.y_test[colname].astype(str).map({'F':0, 'T':1})
        
        self.y_train = self.y_train.idxmax(axis=1)
        self.y_test = self.y_test.idxmax(axis=1)
        
        lbl = LabelEncoder()
        lbl.fit_transform(self.y_train)
        lbl.transform(self.y_test)
        
        return self
    
# '''
# The following preprocessing steps have been done on the test data and training data
# 1) Tokenization - TreebankWordTokenizer has been used. We get inidividual words from a sentence in this step.
# 2) Stop words removal - This step ensures that words which do not provide any context to the data are removed.
# 3) Stemming - Porter stemming has been used. Words are truncated to their roots here.
# 4) Custom stop words  - Additional redundant words found upon data visualization are removed
# '''

    def _preprocess(self, df):
        '''
        Parameters
        --------------
        df:
        type - Pandas dataframe
        
        return
        ---------------
        type - Pandas dataframe
        The dataframe received is tokenized, stemmed and is cleaned of stop words and returned
        '''
        tbt = TreebankWordTokenizer()
        p_stemmer = PorterStemmer()
        en_stop = get_stop_words('en')
        custom_en_stop = []
        custom_en_stop.append(unicode('product_id'))
        custom_en_stop.append(unicode('api_nam'))
        custom_en_stop.append(unicode('user_id'))
        custom_en_stop.append(unicode('task_nam'))
        custom_en_stop.append(unicode('want'))
        custom_en_stop.append(unicode('go'))
        custom_en_stop.append(unicode('hey'))
        custom_en_stop.append(unicode('also'))
        custom_en_stop.append(unicode('ok'))
        
        df = df.apply(lambda row: row.lower())
        df = df.apply(lambda row: tbt.tokenize(row))
        df = df.apply(lambda row: [i for i in row if i not in en_stop])
        df = df.apply(lambda row: [p_stemmer.stem(i) for i in row])
        df = df.apply(lambda row: [i for i in row if i not in custom_en_stop])
        df = df.apply(lambda x: ' '.join(x))
        
        return df
    
    def preprocess(self):
        
        self.X_train = self._preprocess(self.X_train)
        self.X_test = self._preprocess(self.X_test)
        
        return self
    @timeit
    def wordcloud(self):
        '''
        Parameters
        ----------
        Instance of 'Haptik' object.
        
        Function
        ----------
        Displays the word cloud of the the words in the training data
        '''
        textall = reduce(lambda x,y: ' '.join([x,y]), self.X_train) #Wordcloud needs a string
        wordcloud = WordCloud().generate(textall)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        return self
    @timeit
    def classify(self, vect, model):
        
        vect.fit(self.X_train)
        train_dtm = vect.transform(self.X_train)
        test_dtm = vect.transform(self.X_test)
        model.fit(train_dtm, self.y_train)
        
        y_pred_class = model.predict(test_dtm)
        return (accuracy_score(self.y_test, y_pred_class))
        #print (cross_val_score(model, test_dtm, self.y_test, cv = 5).mean())