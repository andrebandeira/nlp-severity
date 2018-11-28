from sys import exit
from pprint import pprint

import numpy as np

from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import unicodedata
import re

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from bs4 import BeautifulSoup


class nlp(object):
    word_index_map = {}
    current_index = 0        
    options = {}
    
    def __init__(self, options = {}):
        self.options = options

        if ('remove_small_words' not in self.options):
            self.options['remove_small_words'] = True

        if ('lemmatize' not in self.options):
            self.options['lemmatize'] = True

        if ('remove_stop_words' not in self.options):
            self.options['remove_stop_words'] = True

        if ('remove_punctuation' not in self.options):
            self.options['remove_punctuation'] = True

        if ('remove_number' not in self.options):
            self.options['remove_number'] = True

        if ('language' not in self.options):
            self.options['language'] = 'portuguese'
        
        self.word_index_map = {}
        self.current_index = 0   
        
    def get_tokens(self, data, key_text = "text"):
        if (self.options['language'] == 'portuguese'):
            lemmatizer = RSLPStemmer()
            lemmatizer = lemmatizer.stem
            stopword = stopwords.words('portuguese')
        else:
            lemmatizer = WordNetLemmatizer()
            lemmatizer = lemmatizer.lemmatize
            stopword = stopwords.words('english')
            
        for item in data:
            text = item[key_text]
            
            if hasattr(text, 'string'):
                text = text.string

            text = text.lower()
            tokens = word_tokenize(text)

            if (self.options['lemmatize']):
                tokens = [lemmatizer(t) for t in tokens]                

            if (self.options['remove_stop_words']):
                tokens = [t for t in tokens if t not in stopword]

            if (self.options['remove_punctuation']):
                tokens = [self.remove_punctuation(t) for t in tokens]

            if (self.options['remove_number']):
                tokens = [self.remove_number(t) for t in tokens]
                
            if (self.options['remove_small_words']):
                tokens = [t for t in tokens if len(t) > 3]

            for token in tokens:
                if token not in self.word_index_map:
                    self.word_index_map[token] = self.current_index
                    self.current_index += 1

            item['tokens'] = tokens

        return data

    def get_data(self, array, key_token = "tokens", key_label = "label"):
        data = np.zeros((len(array), len(self.word_index_map) + 1))
        i = 0
        
        for item in array:
            row = self.tokens_to_numeric(item[key_token], item[key_label])
            data[i,:] = row
            i += 1

        np.random.shuffle(data)
        
        return data
    
    def tokens_to_numeric(self, tokens, label):
        x = np.zeros(len(self.word_index_map) + 1)
        for t in tokens:
            i = self.word_index_map[t]
            x[i] += 1
        x = x / x.sum()
        x[-1] = label
        return x

    def remove_punctuation(self, word):
        nfkd = unicodedata.normalize('NFKD', word)
        word = u"".join([c for c in nfkd if not unicodedata.combining(c)])

        return re.sub('[^a-zA-Z0-9 \\\]', '', word)

    def remove_number(self, word):
        return re.sub('[^a-zA-Z \\\]', '', word)

    def test_data(self, data, length = 100, number_times = 10):
        result = {}
        classifiers = ['LogisticRegression', 'MultinomialNB', 'AdaBoostClassifier', 'SVC', 'LinearSVC', 'SVCScale']

        for classifier in classifiers: 
            result[classifier] = {}
            result[classifier]['scores'] = []
        
        for i in range(number_times):
            np.random.shuffle(data)

            features = data[:,:-1]
            labels = data[:,-1]

            features_train = features[:-length,]
            labels_train = labels[:-length,]

            features_test = features[-length:,]
            labels_test = labels[-length:,]

            for classifier in classifiers:
                model = self.instance_classifier(classifier)
                model.fit(features_train, labels_train)
                result[classifier]['scores'].append(model.score(features_test, labels_test))

        for classifier in classifiers: 
            result[classifier]['avg_score'] = np.mean(result[classifier]['scores'])
            
        return result

    def instance_classifier(self, classifier):
        return {
            'LogisticRegression': LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial'),
            'MultinomialNB': MultinomialNB(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'SVC': svm.SVC(gamma='scale'),
            'LinearSVC': svm.LinearSVC(),
            'SVCScale': svm.SVC(gamma='scale', decision_function_shape='ovo')
        }[classifier]

    def array_merge(self, arrays, balance = True):
        if (not balance):
            return np.concatenate(arrays)

        newArrays = []
        
        length = min(map(len, arrays))
    
        for array in arrays:
            array = array[:length]
            newArrays.append(array)

        return np.concatenate(newArrays)
        
