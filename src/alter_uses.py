from sys import exit
from pprint import pprint
import numpy as np
import random
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nlp import NLP
from uses_multi import USES_MULTI

class ALTER_USES:
        
    def get_dict(features, labels, max_words = 20):
        real_features = features.copy()
        real_labels = labels.copy()

        words_label = USES_MULTI.words_label(features, labels)
        number_words = len(words_label)

        if (number_words < max_words):
            max_words = number_words
            
        features_score = USES_MULTI.feature_score(words_label, labels)        
        
        results = {}
        number_words = 1
        
        while (number_words <= max_words):
            dict_words = ALTER_USES.build_dict(features, labels, features_score, number_words)

            filtered_features = NLP.filter_features(features, dict_words)

            try:
                #pprint(number_words)
                FMeasure = USES_MULTI.classifier(
                    filtered_features,
                    labels
                )
                #pprint(FMeasure)
            except Exception as e:
                #pprint(e)
                FMeasure = 0            

            results.update({
                FMeasure: dict_words
            })
            
            number_words = number_words + 1

        results = sorted(results.items(), key=lambda x: x[0], reverse=True)

        best_FM = results[0][0]
        dict_words = results[0][1]
        
        #print('Best FM: ', best_FM)
        
        return dict_words   
                   

    def build_dict(features, labels, features_score, number_words):
        dict_words = set()
        
        size = len(features)
        for i in range(0,size):
            item = features[i]
            score = features_score[labels[i]]

            size_score = len(score)
            words = []
            for j in range(0,size_score):
                word = score[j][0]
                if word in item:
                    words.append(word)
                    dict_words.add(word)
                    if (len(words) == number_words):
                        break
                    
        return dict_words
