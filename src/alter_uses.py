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
        
    def get_dict(features, labels, max_words = 5):
        real_features = features.copy()
        real_labels = labels.copy()

        words_label = USES_MULTI.words_label(features, labels)
        number_words = len(words_label)

        if (number_words < max_words):
            max_words = number_words
            
        features_score = ALTER_USES.feature_score(words_label, labels)        
        
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

    def feature_score(words_label, labels):
        distinct_labels = list(set(labels))
        score_label = 0;
        score_others_labels = 0
        
        score = {}
        
        for label in distinct_labels:
            others_labels = [t for t in distinct_labels if t != label]
            
            for word in words_label:
                score_label = ALTER_USES.affinity_score(label, word, words_label, labels)
                score_others_labels = 0
                for other in others_labels:
                    score_others_labels += ALTER_USES.affinity_score(other, word, words_label, labels)

                if  label not in score:
                    score[label] = {}
                    
                score[label][word] = score_label - score_others_labels

        
        for label in score:
            score[label] = sorted(score[label].items(), key=lambda x: x[1], reverse=True)
        
        return score            
                
    def affinity_score(label, word, words_label, labels):
        label = str(label)

        if label in words_label[word]:
            defect_word_label = words_label[word][label]
        else:
            defect_word_label = 0

        defect_label =  labels.count(label)
        defect_word  = sum(words_label[word].values())

        #affinity = defect_word_label * (defect_label / defect_word)
        affinity = defect_word_label / defect_word

        #affinity = ((defect_word_label / defect_label) * 0.2) + ((defect_word_label / defect_word) * 0.8)
        return affinity
