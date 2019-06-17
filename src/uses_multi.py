from sys import exit
from pprint import pprint
import numpy as np
import random
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nlp import NLP

class USES_MULTI:
        
    def get_dict(features, labels, percentage_words = 0.1, iterations = 50):
        real_features = features.copy()
        
        words_label = USES_MULTI.words_label(features, labels)

        number_words = len(words_label)
        number_selected_words = int(round(number_words * percentage_words))

        distinct_labels = list(set(labels))
        number_selected_words_label = int(number_selected_words / len(distinct_labels))

        features_score = USES_MULTI.feature_score(words_label, labels)        
        
        candidates = {}      

        for label in features_score:                      
            positive_candidates = features_score[label][:number_selected_words_label]
            candidates.update(positive_candidates)
            negative_candidates = features_score[label][-number_selected_words_label:]  
            candidates.update(negative_candidates)
                    
        results = {}
        actual_iteration = 0
        
        while (actual_iteration < iterations):            
            actual_candidates = USES_MULTI.random_items(candidates, number_selected_words)
            filtered_features = NLP.filter_features(features, actual_candidates)
            try:
                #pprint(actual_iteration)
                FMeasure = USES_MULTI.classifier(
                    filtered_features,
                    labels
                )
                #pprint(FMeasure)
            except Exception as e:
                #pprint(e)
                FMeasure = 0            

            results.update({
                FMeasure: actual_candidates
            })
            
            actual_iteration = actual_iteration + 1

        results = sorted(results.items(), key=lambda x: x[0], reverse=True)

        best_FM = results[0][0]
        dict_words = results[0][1]
        
        #print('Best: ', best_FM)
        
        return dict_words
        
    def words_label(features, labels):
        words_label = {}
        
        size = len(features)
        for i in range(0,size):
            item = features[i]
            label = labels[i]
            label = str(label)
                            
            sizeItem = len(item)
            for j in range(0,sizeItem):
                feature = item[j]

                if feature not in words_label:
                    words_label[feature] = {}
                    
                if label not in words_label[feature]:
                    words_label[feature][label] = 0

                words_label[feature][label] = words_label[feature][label] + 1

        return words_label
        
    def feature_score(words_label, labels):
        distinct_labels = list(set(labels))
        score_label = 0;
        score_others_labels = 0
        
        score = {}
        
        for label in distinct_labels:
            others_labels = [t for t in distinct_labels if t != label]
            
            for word in words_label:
                score_label = USES_MULTI.affinity_score(label, word, words_label, labels)
                score_others_labels = 0
                for other in others_labels:
                    score_others_labels += USES_MULTI.affinity_score(other, word, words_label, labels)

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

        affinity = defect_word_label / (defect_label + defect_word - defect_word_label)
        
        return affinity

    def random_items(items, number):
        if (len(items) < number):
            number = len(items)
            
        items_copy = items.copy()
        
        new_items = {}
        
        keys = random.sample(list(items_copy), number)
        
        for key in keys:
            new_items[key] = items_copy[key]

        return new_items
                   
    def classifier(features, labels):
        features_copy = features.copy()
        labels_copy = labels.copy()
        
        features_copy = NLP.text_to_numeric(features_copy)

        results = NLP.test(features_copy, labels)
        betterFM = 0

        for classifier in results:
            metrics = results[classifier];
            if (metrics['F1']['avg'] > betterFM):
                betterFM= metrics['F1']['avg']

        return betterFM;





    
    
    

        
                
            
        
        

  
