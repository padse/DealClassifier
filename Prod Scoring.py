#!/usr/bin/python -tt
import pandas as pd
import nltk.stem
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
#from nltk import stem  

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
    


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def accuracy(true, pred):
    return (true == pred).sum() / float(true.shape[0])

def macro(true, pred):
    scores = []
    for l in np.unique(true):
        scores.append(accuracy(np.where(true != l, 1, 0),
                               np.where(pred != l, 1, 0)))
    return float(sum(scores)) / float(len(scores))

def main():
    try:

        
        ADS = pd.read_csv(r'./ADS_S.csv')       
        ADS = ADS[ADS.Category != 'Prime Pantry']
        from sklearn.cross_validation import train_test_split
        text_train, text_test, y_train, y_test = train_test_split(ADS.title, ADS.Category, random_state=100, test_size=0.30,\
                                                              stratify=ADS.Category)

        
        ############################## 
        #y_test =
        #text_test =
        ##############################
        #Define Pandas series with product title and y value here.By default it points to the test data set from model development
        
        loaded_pipeline = joblib.load('./Log_models.pkl')
        pred_val = loaded_pipeline.predict(text_test)

        print classification_report(y_test, pred_val)
        print "****************"
        print('accuracy:', accuracy(y_test, pred_val))
        print "****************"
        print('average-per-class accuracy:', macro(y_test, pred_val))

    except Exception as e:
        print e


if __name__ == '__main__':
  main()

