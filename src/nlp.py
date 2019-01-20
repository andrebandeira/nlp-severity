from sys import exit
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

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

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

import random

class NLP:
    def tokenizer(features):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = item.strip()
            item = item.lower()
            item = word_tokenize(item)
            features[i] = item
            
        return features

    def remove_numbers(features):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [NLP.number_remove(t) for t in item]
            item = list(filter(None, item))
            features[i] = item
        
        return features

    @staticmethod 
    def number_remove(word):
        return re.sub('[0-9\\\]', '', word)
    
    def remove_small_words(features, min_length = 3):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [t for t in item if len(t) >= min_length]
            features[i] = item
          
        return features

    def remove_stop_words(features, language = 'english'):
        if (language == 'portuguese'):
            stopword = stopwords.words('portuguese')
        else:
            stopword = stopwords.words('english')
            
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [t for t in item if t not in stopword]
            features[i] = item
               
        return features

    def  lemmatizer(features, language = 'english'):
        if (language == 'portuguese'):
            lemmatizer = RSLPStemmer().stem
        else:
            lemmatizer = WordNetLemmatizer().lemmatize

        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [lemmatizer(t) for t in item]
            item = list(filter(None, item))
            features[i] = item

        return features

    def remove_punctuation(features):
        size = len(features)
        for i in range(0,size):            
            item = features[i]
            item = [NLP.punctuation_remove(t) for t in item]
            item = list(filter(None, item))
            features[i] = item
        
        return features

    
    @staticmethod 
    def punctuation_remove(word):
        nfkd = unicodedata.normalize('NFKD', word)
        word = u"".join([c for c in nfkd if not unicodedata.combining(c)])        
        return re.sub('[^a-zA-Z0-9 \\\]', '', word)

    def text_to_numeric(features, method = 'tf_idf'):
        def do_nothing(doc):
            return doc
        
        tfidf = TfidfVectorizer(
            analyzer='word',
            tokenizer=do_nothing,
            preprocessor=do_nothing,
            token_pattern=None)

        tfidf.fit(features)

        if (method == 'tf_idf'):
            return NLP.tf_idf(tfidf, features)
        elif (method == 'tf_normalize'):
            return NLP.tf_normalize(tfidf, features)
        elif (method == 'tf'):
            return NLP.tf(tfidf, features)
        elif (method == 'binary'):
            return NLP.binary(tfidf, features)

        
    @staticmethod 
    def tf_idf(tfidf, features):       
        return tfidf.transform(features).todense()

    @staticmethod
    def tf_normalize(tfidf, features):
        size = len(features)
        
        word_map = tfidf.vocabulary_;

        data = np.zeros((size, len(word_map)))

        def tokens_to_numeric(tokens):
            x = np.zeros(len(word_map))
        
            for t in tokens:
                if t in word_map:
                    i = word_map[t]
                    x[i] += 1
                    
            x_sum = x.sum()
            
            if (x_sum > 0):
                x = x / x_sum            
            
            return x
            
        for i in range(0,size):
            tokens = features[i]            
            row = tokens_to_numeric(tokens)
            data[i,:] = row

        return data;

    def tf(tfidf, features):
        size = len(features)
        
        word_map = tfidf.vocabulary_;

        data = np.zeros((size, len(word_map)))

        def tokens_to_numeric(tokens):
            x = np.zeros(len(word_map))
        
            for t in tokens:
                if t in word_map:
                    i = word_map[t]
                    x[i] += 1          
            
            return x
            
        for i in range(0,size):
            tokens = features[i]            
            row = tokens_to_numeric(tokens)
            data[i,:] = row

        return data;

    @staticmethod
    def binary(tfidf, features):
        size = len(features)
        
        word_map = tfidf.vocabulary_;

        data = np.zeros((size, len(word_map)))

        def tokens_to_numeric(tokens):
            x = np.zeros(len(word_map))
        
            for t in tokens:
                if t in word_map:
                    i = word_map[t]
                    x[i] = 1          
            
            return x
            
        for i in range(0,size):
            tokens = features[i]            
            row = tokens_to_numeric(tokens)
            data[i,:] = row

        return data;

    def dim_reduction(features, method= 'LSA', n_features = 500):
        if (method == 'LSA'):
            redu = TruncatedSVD(n_components=n_features)
        elif (method == 'PCA'):
            redu = PCA(n_components=n_features)

        return redu.fit_transform(features)
    
    @staticmethod
    def test(features, labels, folds = 10, classifiers = []):
        result = {}
        
        if (len(classifiers) == 0):
            classifiers = [
                #'LogisticRegression',
                'MultinomialNB',
                #'AdaBoostClassifier',
                #'SVC',
                'LinearSVC',
                #'SVCScale',
                #'DecisionTree',
                #'RandomForest'
            ]

        for classifier in classifiers:
            try:
                model = NLP.get_classifier(classifier)
                result[classifier] = NLP.calc_cross_validate(model, features, labels, folds)
            except:
                print ("Erro ao executar classificador: ", classifier)
                
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
        
        result.update(NLP.get_metric('Accuracy', scores['test_accuracy']))
        result.update(NLP.get_metric('Precision', scores['test_precision_macro']))
        result.update(NLP.get_metric('Recall', scores['test_recall_macro']))
        result.update(NLP.get_metric('F1', scores['test_f1_macro']))

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
        newArrays = []
        
        if (not balance):
            newArrays = np.concatenate(arrays)
        else:
            length = min(map(len, arrays))
        
            for array in arrays:
                array = array[:length]
                newArrays.append(array)

            newArrays = np.concatenate(newArrays)

        np.random.shuffle(newArrays)
        return newArrays


