from sys import exit
from pprint import pprint

import numpy as np

from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import unicodedata
import re

from sklearn.datasets import load_iris

from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

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
        
    def test_data(self, data, folds = 10, type_calc = 'validate'):
        result = {}
        
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

        #classifiers = ['LinearSVC']

        features = data[:,:-1]
        real_labels = data[:,-1]

        #data = load_iris()
        #data = load_linnerud()

        #features = data.data
        #real_labels = data.target         
        
        for classifier in classifiers:
            model = self.instance_classifier(classifier)

            if (type_calc == 'validate'):
                result[classifier] = self.calc_cross_validate(model, features, real_labels, folds)
            elif (type_calc == 'predict'):
                result[classifier] = self.calc_cross_val_predict(model, features, real_labels, folds)
            elif (type_calc == 'manual'):
                result[classifier] = self.calc_manual(model, features, real_labels, folds)            
            
        return result

    def instance_classifier(self, classifier):
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

    def array_merge(self, arrays, balance = True):
        if (not balance):
            return np.concatenate(arrays)

        newArrays = []
        
        length = min(map(len, arrays))
    
        for array in arrays:
            array = array[:length]
            newArrays.append(array)

        return np.concatenate(newArrays)

    def calc_manual(self, model, features, real_labels, folds):       
        result = {}

        length = int(len(features) / folds)

        train_features = features[:-length,]
        train_labels = real_labels[:-length,]

        test_features = features[-length:,]
        test_labels = real_labels[-length:,]

        classifier = model.fit(train_features, train_labels)

        pred_labels = classifier.predict(test_features)

        result['Accuracy'] = {}
        result['Precision'] = {}
        result['Recall'] = {}
        result['F1'] = {}

        result['Accuracy']['avg'] = accuracy_score(test_labels, pred_labels)
        result['Precision']['avg'] = precision_score(test_labels, pred_labels, average = 'macro')
        result['Recall']['avg'] = recall_score(test_labels, pred_labels, average = 'macro')
        result['F1']['avg'] = f1_score(test_labels, pred_labels, average = 'macro')

        return result
    
    def calc_cross_val_predict(self, model, features, real_labels, folds):
        result = {}
        
        cv = StratifiedKFold(n_splits=folds)

        pred_labels = cross_val_predict(
            model,
            features,
            real_labels,
            cv=cv
        )

        result['Accuracy'] = {}
        result['Precision'] = {}
        result['Recall'] = {}
        result['F1'] = {}

        result['Accuracy']['avg'] = accuracy_score(real_labels, pred_labels)
        result['Precision']['avg'] = precision_score(real_labels, pred_labels, average = 'macro')
        result['Recall']['avg'] = recall_score(real_labels, pred_labels, average = 'macro')
        result['F1']['avg'] = f1_score(real_labels, pred_labels, average = 'macro')

        return result

    def calc_cross_validate(self, model, features, real_labels, folds):
        result = {}
        
        cv = StratifiedKFold(n_splits=folds)
        
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        scores = cross_validate(
            model,
            features,
            real_labels,
            scoring=scoring,
            cv=cv
        )
        
        result.update(self.get_metric('Accuracy', scores['test_accuracy']))
        result.update(self.get_metric('Precision', scores['test_precision_macro']))
        result.update(self.get_metric('Recall', scores['test_recall_macro']))
        result.update(self.get_metric('F1', scores['test_f1_macro']))

        return result

    def get_metric(self, metric, values):
        result = {}
        result[metric] = {}

        result[metric]['values'] = values
        result[metric]['avg'] = values.mean()
        result[metric]['max'] = values.max()
        result[metric]['min'] = values.min()
        result[metric]['median'] = np.median(values)
        result[metric]['std'] = values.std()

        return result
        
