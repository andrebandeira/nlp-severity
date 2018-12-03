from sys import exit
from pprint import pprint

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer

import numpy as np
import re
import unicodedata

class Tokenizer(FunctionTransformer):
                
    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        size = len(corpus)
        for i in range(0,size):
            item = corpus[i]
            if (isinstance(item, str)):
                item = item.strip()
                item = item.lower()
                item = word_tokenize(item)
                corpus[i] = item
            
        return corpus

class Remove_Numbers(FunctionTransformer):
        
    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        size = len(corpus)
        for i in range(0,size):
            item = corpus[i]
            item = [Remove_Numbers.remove(t) for t in item]
            item = list(filter(None, item))
            corpus[i] = item
        
        return corpus

    @staticmethod 
    def remove(word):
        return re.sub('[0-9\\\]', '', word) 

class Remove_Small_Words(FunctionTransformer):

    min_length = {}
    
    def __init__(self, min_length = 3):
        self.min_length = min_length
        
    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        size = len(corpus)
        for i in range(0,size):
            item = corpus[i]
            item = [t for t in item if len(t) > self.min_length]
            corpus[i] = item
          
        return corpus

class Remove_Stop_Words(FunctionTransformer):

    stopword = {}
    
    def __init__(self, language = 'english'):
        if (language == 'portuguese'):
            self.stopword = stopwords.words('portuguese')
        else:
            self.stopword = stopwords.words('english')
                    
    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        size = len(corpus)
        for i in range(0,size):
            item = corpus[i]
            item = [t for t in item if t not in self.stopword]
            corpus[i] = item
               
        return corpus

class Lemmatizer(FunctionTransformer):

    lemmatizer = {}
    
    def __init__(self, language = 'english'):
        if (language == 'portuguese'):
            self.lemmatizer = RSLPStemmer().stem
        else:
            self.lemmatizer = WordNetLemmatizer().lemmatize
            
    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        size = len(corpus)
        for i in range(0,size):
            item = corpus[i]
            item = [self.lemmatizer(t) for t in item]
            item = list(filter(None, item))
            corpus[i] = item

        return corpus

class Remove_Punctuation(FunctionTransformer):
        
    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        size = len(corpus)
        for i in range(0,size):            
            item = corpus[i]
            item = [Remove_Punctuation.remove(t) for t in item]
            item = list(filter(None, item))
            corpus[i] = item
        
        return corpus

    @staticmethod 
    def remove(word):
        nfkd = unicodedata.normalize('NFKD', word)
        word = u"".join([c for c in nfkd if not unicodedata.combining(c)])        
        return re.sub('[^a-zA-Z0-9 \\\]', '', word)

class Text_To_Numeric(FunctionTransformer):
    word_map = {}
    word_map_defined = False
    
    def fit(self, X, y=None):
        return self

    def transform(self, corpus):       
        size = len(corpus)
        
        Text_To_Numeric.define_word_map(corpus)
        
        data = np.zeros((size, len(Text_To_Numeric.word_map)))
        
        for i in range(0,size):
            tokens = corpus[i]            
            row = Text_To_Numeric.tokens_to_numeric(tokens)
            data[i,:] = row

        return data

    @staticmethod 
    def define_word_map(corpus):
        if (Text_To_Numeric.word_map_defined == False):
            temp_word_map = {}
            current_index = 0
            size = len(corpus)
            for i in range(0,size):
                tokens = corpus[i]
                for token in tokens:
                    if token not in temp_word_map:
                        temp_word_map[token] = current_index
                        current_index += 1

            Text_To_Numeric.word_map = temp_word_map
            Text_To_Numeric.word_map_defined = True                
        

    @staticmethod     
    def tokens_to_numeric(tokens):
        x = np.zeros(len(Text_To_Numeric.word_map))
        
        for t in tokens:
            if t in Text_To_Numeric.word_map:
                i = Text_To_Numeric.word_map[t]
                x[i] += 1
                
        x_sum = x.sum()
        
        if (x_sum > 0):
            x = x / x_sum            
        
        return x

class Utils(object):
    @staticmethod
    def test(features, labels, folds = 10, classifiers = []):
        result = {}
        
        if (len(classifiers) == 0):
            classifiers = [
                'LogisticRegression',
                'MultinomialNB',
                'AdaBoostClassifier',
                'SVC',
                'LinearSVC',
                'SVCScale',
                'DecisionTree',
                'RandomForest'
            ]

        for classifier in classifiers:
            model = Utils.get_classifier(classifier)
            result[classifier] = Utils.calc_cross_validate(model, features, labels, folds)

        return result

    @staticmethod
    def get_classifier(classifier):
        return {
            'LogisticRegression': LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial'),
            'MultinomialNB': MultinomialNB(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'SVC': SVC(gamma='scale'),
            'LinearSVC': LinearSVC(max_iter=10000),
            'SVCScale': SVC(gamma='scale', decision_function_shape='ovo'),
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        }[classifier]

    @staticmethod
    def calc_cross_validate(model, features, labels, folds):
        result = {}
               
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        scores = cross_validate(
            model,
            features,
            labels,
            scoring=scoring,
            cv=folds
        )
        
        result.update(Utils.get_metric('Accuracy', scores['test_accuracy']))
        result.update(Utils.get_metric('Precision', scores['test_precision_macro']))
        result.update(Utils.get_metric('Recall', scores['test_recall_macro']))
        result.update(Utils.get_metric('F1', scores['test_f1_macro']))

        return result

    @staticmethod
    def get_metric(metric, values):
        result = {}
        result[metric] = {}

        result[metric]['values'] = values
        result[metric]['avg'] = values.mean()
        result[metric]['max'] = values.max()
        result[metric]['min'] = values.min()
        result[metric]['median'] = np.median(values)
        result[metric]['std'] = values.std()

        return result
    
    @staticmethod
    def array_merge(arrays, balance = True):
        if (not balance):
            return np.concatenate(arrays)

        newArrays = []
            
        length = min(map(len, arrays))
        
        for array in arrays:
            array = array[:length]
            newArrays.append(array)

        return np.concatenate(newArrays)        


